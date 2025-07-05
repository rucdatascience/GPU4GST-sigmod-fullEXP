#ifndef _ENACTOR_H_
#define _ENACTOR_H_
#include "gpu_graph.cuh"
#include "meta_data.cuh"

#include "util.h"
#include "header.h"
#include "prefix_scan.cuh"
#include <limits.h>
#include "barrier.cuh"

__global__ void
balanced_push_kernel(
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier)
{

	//__shared__ vertex_t smem[32]; // 32位共享内存
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;   // 线程在warp的位置
	const index_t wid_in_blk = threadIdx.x >> 5;   // warp在block的位置
	const index_t wid_in_grd = TID >> 5;		   // warp在grid的位置
	const index_t wcount_in_blk = blockDim.x >> 5; // 一个block里面有多少warp
	const index_t WGRNTY = GRNTY >> 5;			   // warp stride

	feature_t level_thd = level[0];

	if (!TID)
	{
		level[6] = 0;
		mdata.worklist_sz_mid[0] = 0;
	}

	global_barrier.sync_grid_opt();

	int r = 0;
	while (true && ++r < 20)
	{
		// 平衡
		mdata.future_work[0] = 0;
		global_barrier.sync_grid_opt();

		if (!TID)
		{
			int tot = mdata.worklist_sz_mid[0] + mdata.worklist_sz_sml[0] + mdata.worklist_sz_lrg[0];
			level[6] += tot;
			mdata.worklist_sz_mid[0] = 0; // 下面应该是在扫描执行了 先把新的队列长度置0
			mdata.worklist_sz_sml[0] = 0; // indicate whether bin overflow
			mdata.worklist_sz_lrg[0] = 0;
			if (tot > level[5])
				level[5] = tot;
		}

		worklist_gather._push_coalesced_scan_random_list_atomic(TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd + 1);

		global_barrier.sync_grid_opt();
		if (mdata.worklist_sz_sml[0] +
				mdata.worklist_sz_mid[0] +
				mdata.worklist_sz_lrg[0] ==
			0)
			break;

		compute_mapper.mapper_push(
			mdata.worklist_sz_lrg[0],
			mdata.worklist_lrg,
			mdata.cat_thd_count_lrg,
			blockIdx.x,	 /*group id*/
			blockDim.x,	 /*group size*/
			gridDim.x,	 /*group count*/
			threadIdx.x, /*thread off intra group*/
			level_thd);

		compute_mapper.mapper_push(
			mdata.worklist_sz_mid[0],
			mdata.worklist_mid,
			mdata.cat_thd_count_mid,

			wid_in_grd, /*group id*/
			32,			/*group size*/
			WGRNTY,		/*group count*/
			tid_in_wrp, /*thread off intra group*/
			level_thd);

		compute_mapper.mapper_push(
			mdata.worklist_sz_sml[0],
			mdata.worklist_sml,
			mdata.cat_thd_count_sml,
			TID,   /*group id*/
			1,	   /*group size*/
			GRNTY, /*group count*/
			0,	   /*thread off intra group*/
			level_thd);

		//      global_barrier.sync_grid_opt();

		_grid_sum<vertex_t, index_t>(mdata.cat_thd_count_sml[TID] +
										 mdata.cat_thd_count_mid[TID] +
										 mdata.cat_thd_count_lrg[TID],
									 mdata.future_work);

		global_barrier.sync_grid_opt();
		if (mdata.future_work[0] > ggraph.edge_count * SWITCH_TO && 0)
		{

			break;
		}
	}

	if (!TID)
		level[0] = level_thd;
}

int balanced_push(
	int cfg_blk_size,
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier)
{
	// 分成三种情况大中小 现在要改的函数！！！
	int blk_size = 0;
	int grd_size = 0;
	// cudaFuncGetAttributes
	cudaOccupancyMaxPotentialBlockSize(&grd_size, &blk_size,
									   balanced_push_kernel, 0, 0);

	grd_size = (blk_size * grd_size) / cfg_blk_size;
	blk_size = cfg_blk_size;

	assert(blk_size * grd_size <= BLKS_NUM * THDS_NUM);

	// push_pull_opt_kernel
	double time = wtime();
	balanced_push_kernel<<<grd_size, blk_size>>>(level,
												 ggraph,
												 mdata,
												 compute_mapper,
												 worklist_gather,
												 global_barrier);

	H_ERR(cudaDeviceSynchronize());

	return 0;
}

__global__ void balanced_push_kernel_atomic(
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier)
{

	//__shared__ vertex_t smem[32]; // 32位共享内存
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;   // 线程在warp的位置
	const index_t wid_in_blk = threadIdx.x >> 5;   // warp在block的位置
	const index_t wid_in_grd = TID >> 5;		   // warp在grid的位置
	const index_t wcount_in_blk = blockDim.x >> 5; // 一个block里面有多少warp
	const index_t WGRNTY = GRNTY >> 5;			   // warp stride

	feature_t level_thd = level[0];

	if (!TID)
	{
		level[6] = 0;
		mdata.worklist_sz_mid[0] = 0;
	}

	global_barrier.sync_grid_opt();

	int r = 0;
	while (true && ++r < 20)
	{
		// 平衡
		mdata.future_work[0] = 0;
		global_barrier.sync_grid_opt();

		if (!TID)
		{
			int tot = mdata.worklist_sz_mid[0] + mdata.worklist_sz_sml[0] + mdata.worklist_sz_lrg[0];
			level[6] += tot;
			mdata.worklist_sz_mid[0] = 0; // 下面应该是在扫描执行了 先把新的队列长度置0
			mdata.worklist_sz_sml[0] = 0; // indicate whether bin overflow
			mdata.worklist_sz_lrg[0] = 0;
			if (tot > level[5])
				level[5] = tot;
		}

		worklist_gather._push_coalesced_scan_random_list_atomic(TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd + 1);

		global_barrier.sync_grid_opt();
		if (mdata.worklist_sz_sml[0] +
				mdata.worklist_sz_mid[0] +
				mdata.worklist_sz_lrg[0] ==
			0)
			break;

		compute_mapper.mapper_push(
			mdata.worklist_sz_lrg[0],
			mdata.worklist_lrg,
			mdata.cat_thd_count_lrg,
			blockIdx.x,	 /*group id*/
			blockDim.x,	 /*group size*/
			gridDim.x,	 /*group count*/
			threadIdx.x, /*thread off intra group*/
			level_thd);

		compute_mapper.mapper_push(
			mdata.worklist_sz_mid[0],
			mdata.worklist_mid,
			mdata.cat_thd_count_mid,

			wid_in_grd, /*group id*/
			32,			/*group size*/
			WGRNTY,		/*group count*/
			tid_in_wrp, /*thread off intra group*/
			level_thd);

		compute_mapper.mapper_push(
			mdata.worklist_sz_sml[0],
			mdata.worklist_sml,
			mdata.cat_thd_count_sml,
			TID,   /*group id*/
			1,	   /*group size*/
			GRNTY, /*group count*/
			0,	   /*thread off intra group*/
			level_thd);

		//      global_barrier.sync_grid_opt();

		_grid_sum<vertex_t, index_t>(mdata.cat_thd_count_sml[TID] +
										 mdata.cat_thd_count_mid[TID] +
										 mdata.cat_thd_count_lrg[TID],
									 mdata.future_work);

		global_barrier.sync_grid_opt();
		if (mdata.future_work[0] > ggraph.edge_count * SWITCH_TO && 0)
		{

			break;
		}
	}

	if (!TID)
		level[0] = level_thd;
}

int balanced_push_atomic(
	int cfg_blk_size,
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier)
{
	// 分成三种情况大中小 现在要改的函数！！！
	int blk_size = 0;
	int grd_size = 0;
	// cudaFuncGetAttributes
	cudaOccupancyMaxPotentialBlockSize(&grd_size, &blk_size,
									   balanced_push_kernel, 0, 0);

	grd_size = (blk_size * grd_size) / cfg_blk_size;
	blk_size = cfg_blk_size;

	assert(blk_size * grd_size <= BLKS_NUM * THDS_NUM);

	// push_pull_opt_kernel
	double time = wtime();
	balanced_push_kernel_atomic<<<grd_size, blk_size>>>(level,
												 ggraph,
												 mdata,
												 compute_mapper,
												 worklist_gather,
												 global_barrier);

	H_ERR(cudaDeviceSynchronize());

	return 0;
}
__global__ void
balanced_push_kernel_sml(
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier
	)
{

//__shared__ vertex_t smem[32]; // 32位共享内存
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;   // 线程在warp的位置
	const index_t wid_in_blk = threadIdx.x >> 5;   // warp在block的位置
	const index_t wid_in_grd = TID >> 5;		   // warp在grid的位置
	const index_t wcount_in_blk = blockDim.x >> 5; // 一个block里面有多少warp
	const index_t WGRNTY = GRNTY >> 5;			   // warp stride

	feature_t level_thd = level[0];;

	// and generate frontiers immediately
	global_barrier.sync_grid_opt();
	// global_barrier.sync_grid_opt();
	// Three push mappers.

		compute_mapper.mapper_push_new(
			mdata.worklist_sz_sml[0],
			mdata.worklist_sml,
			mdata.cat_thd_count_sml,
			TID,   /*group id*/
			1,	   /*group size*/
			GRNTY, /*group count*/
			0,	   /*thread off intra group*/
			level_thd);

	//      global_barrier.sync_grid_opt();

	global_barrier.sync_grid_opt();
}

__global__ void
balanced_push_kernel_mid(
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier
	)
{
//__shared__ vertex_t smem[32]; // 32位共享内存
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;   // 线程在warp的位置
	const index_t wid_in_blk = threadIdx.x >> 5;   // warp在block的位置
	const index_t wid_in_grd = TID >> 5;		   // warp在grid的位置
	const index_t wcount_in_blk = blockDim.x >> 5; // 一个block里面有多少warp
	const index_t WGRNTY = GRNTY >> 5;			   // warp stride

	feature_t level_thd = level[0];;

	// and generate frontiers immediately
	global_barrier.sync_grid_opt();
	// global_barrier.sync_grid_opt();
	// Three push mappers.

	compute_mapper.mapper_push_new(
		mdata.worklist_sz_mid[0],
		mdata.worklist_mid,
		mdata.cat_thd_count_mid,

		wid_in_grd, /*group id*/
		32,			/*group size*/
		WGRNTY,		/*group count*/
		tid_in_wrp, /*thread off intra group*/
		level_thd);

	//      global_barrier.sync_grid_opt();

	global_barrier.sync_grid_opt();

}

__global__ void
balanced_push_kernel_lrg(
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier)
{
//__shared__ vertex_t smem[32]; // 32位共享内存
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;   // 线程在warp的位置
	const index_t wid_in_blk = threadIdx.x >> 5;   // warp在block的位置
	const index_t wid_in_grd = TID >> 5;		   // warp在grid的位置
	const index_t wcount_in_blk = blockDim.x >> 5; // 一个block里面有多少warp
	const index_t WGRNTY = GRNTY >> 5;			   // warp stride

	feature_t level_thd = level[0];;

	// and generate frontiers immediately
	global_barrier.sync_grid_opt();
	// global_barrier.sync_grid_opt();
	// Three push mappers.


		compute_mapper.mapper_push_new(
			mdata.worklist_sz_lrg[0],
			mdata.worklist_lrg,
			mdata.cat_thd_count_lrg,
			blockIdx.x,	 /*group id*/
			blockDim.x,	 /*group size*/
			gridDim.x,	 /*group count*/
			threadIdx.x, /*thread off intra group*/
			level_thd);
	//      global_barrier.sync_grid_opt();

	global_barrier.sync_grid_opt();
}

__global__ void
collect_work(
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier,
	int *work_size_sml,
	int *work_size_mid,
	int *work_size_lrg)
{
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;   // 线程在warp的位置
	const index_t wid_in_blk = threadIdx.x >> 5;   // warp在block的位置
	const index_t wid_in_grd = TID >> 5;		   // warp在grid的位置
	const index_t wcount_in_blk = blockDim.x >> 5; // 一个block里面有多少warp
	feature_t level_thd = level[0];
	if (!TID)
	{
		mdata.worklist_sz_mid[0] = 0; // 下面应该是在扫描执行了 先把新的队列长度置0
		mdata.worklist_sz_sml[0] = 0;
		mdata.worklist_sz_lrg[0] = 0;
	}
	global_barrier.sync_grid_opt();
	worklist_gather._push_coalesced_scan_random_list_atomic(TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd + 1);
	global_barrier.sync_grid_opt();
	if (!TID)
	{
		*work_size_sml = mdata.worklist_sz_sml[0];
		*work_size_mid = mdata.worklist_sz_mid[0];
		
		*work_size_lrg = mdata.worklist_sz_lrg[0];
	}
	global_barrier.sync_grid_opt();
}
int balanced_push_without_fusion(
	int cfg_blk_size,
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier)
{
	// 分成三种情况大中小
	int blk_size = 0;
	int grd_size = 0;
	int *work_size_sml, *work_size_mid, *work_size_lrg;
	cudaMallocManaged((void **)&work_size_sml, sizeof(int));
	cudaMallocManaged((void **)&work_size_mid, sizeof(int));
	cudaMallocManaged((void **)&work_size_lrg, sizeof(int));
	// cudaFuncGetAttributes
	cudaOccupancyMaxPotentialBlockSize(&grd_size, &blk_size,
									   collect_work, 0, 0);

	grd_size = (blk_size * grd_size) / cfg_blk_size;
	blk_size = cfg_blk_size;
	// grd_size = (blk_size * grd_size)/ 128;
	// blk_size = 128;

	// printf("balanced push-- block=%d, grid=%d\n", blk_size, grd_size);
	assert(blk_size * grd_size <= BLKS_NUM * THDS_NUM);
	cout << blk_size << " " << grd_size << endl;
	// push_pull_opt_kernel
	double time = wtime();
	int work_size = 0;
	int rounds = 0;
	collect_work<<<grd_size, blk_size>>>(level,
										 ggraph,
										 mdata,
										 compute_mapper,
										 worklist_gather,
										 global_barrier, work_size_sml, work_size_mid, work_size_lrg);
	cudaDeviceSynchronize();
	//cout << "size " << *work_size_sml << " " << *work_size_mid << " " << *work_size_lrg << endl;
	work_size = *work_size_sml + *work_size_mid + *work_size_lrg;
	while (work_size != 0)
	{
		//rounds++;
		balanced_push_kernel_sml<<<grd_size, blk_size>>>(level,
														 ggraph,
														 mdata,
														 compute_mapper,
														 worklist_gather,
														 global_barrier);
		balanced_push_kernel_mid<<<grd_size, blk_size>>>(level,
														 ggraph,
														 mdata,
														 compute_mapper,
														 worklist_gather,
														 global_barrier);
		cudaDeviceSynchronize();
		balanced_push_kernel_lrg<<<grd_size, blk_size>>>(level,
														 ggraph,
														 mdata,
														 compute_mapper,
														 worklist_gather,
														 global_barrier);
		cudaDeviceSynchronize();

		collect_work<<<grd_size, blk_size>>>(level,
											 ggraph,
											 mdata,
											 compute_mapper,
											 worklist_gather,
											 global_barrier, work_size_sml, work_size_mid, work_size_lrg);
		cudaDeviceSynchronize();
		//cout << "size " << *work_size_sml << " " << *work_size_mid << " " << *work_size_lrg << endl;
		work_size = *work_size_sml + *work_size_mid + *work_size_lrg;
	}
	cudaDeviceSynchronize();
	return 0;
}

#endif
