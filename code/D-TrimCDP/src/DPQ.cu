#include <chrono>
#include <DPQ.cuh>
#include <thrust/device_vector.h>
using namespace std;
typedef unsigned int uint;


struct queue_element_d
{
	uint v, p, d;

    queue_element_d(uint _v = 0, uint _p = 0, uint _d = 0)
        : v(_v), p(_p), d(_d) {}
};

void set_max_ID(graph_v_of_v_idealID &group_graph, std::vector<uint> &cumpulsory_group_vertices, uint *host_tree, std::unordered_set<uint> &contain_group_vertices,uint val1,uint val2)
{
	uint bit_num = 1, v;
	for (auto it = cumpulsory_group_vertices.begin(); it != cumpulsory_group_vertices.end(); it++, bit_num <<= 1)
	{
		for (size_t to = 0; to < group_graph[*it].size(); to++)
		{
			v = group_graph[*it][to].first;
			host_tree[v*val1+ bit_num * val2] = 0;
			contain_group_vertices.insert(v);
		}
	}
}
int get_max(uint vertex, uint *host_tree,uint width, uint val1,uint val2)
{
	int re = 0;
	for (size_t i = 1; i < width; i <<= 1)
	{
		if (host_tree[vertex*val1+ i * val2] == 0)
		{
			re += i;
		}
	}

	return re;
}
inline uint graph_v_of_v_idealID_DPBF_vertex_group_set_ID_gpu(uint vertex, graph_v_of_v_idealID &group_graph,
													  std::unordered_set<uint> &cumpulsory_group_vertices)
{

	/*time complexity: O(|Gamma|); this function returns the maximum group set ID for a single vertex*/
	// if group i have edge to v,v will give bit i value 1;
	uint ID = 0;
	uint pow_num = 0;
	for (auto it = cumpulsory_group_vertices.begin(); it != cumpulsory_group_vertices.end(); it++)
	{
		if (graph_v_of_v_idealID_contain_edge(group_graph, vertex, *it))
		{ // vertex is in group *it
			ID = ID + (1 << pow_num);
		}
		pow_num++;
	}

	return ID;
}

__global__ void Relax(
    int *pointer, int *edge, int *edge_weight,
    queue_element_d *queue, uint *queue_size,
    queue_element_d *queue2, uint *queue_size2,
    uint *host_tree, uint *updated, 
    uint VAL1, uint VAL2, uint group_sets_ID_range, uint D,int *tree_weight) 
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *queue_size) {
        queue_element_d now = queue[idx];
        uint v = now.v;
        uint p = now.p;
        uint d = now.d;
        if (p == group_sets_ID_range) {
            if(*tree_weight == host_tree[v * VAL1 + p * VAL2 + d] )
            {
                *tree_weight = -1;
            }
            return;
        }
        //grow
        uint cost = host_tree[v * VAL1 + p * VAL2 + d];
        if (d < D) {
            for (int i = pointer[v]; i < pointer[v + 1]; i++) {
                uint u = edge[i];
                uint z = edge_weight[i];
                uint id = u * VAL1 + p * VAL2 + (d + 1);
                uint old = atomicMin(&host_tree[id], cost + z);
                if (old > cost + z) {
                    
                    uint t = atomicCAS(&updated[id], 0, 1);
                    if (!t) {
                        uint now = atomicAdd(queue_size2, 1);
                        queue2[now].v = u;
                        queue2[now].p = p;
                        queue2[now].d = d + 1;
                    }
                }
            }
        }
        //merge
        uint p1 = p, d1 = d;
        uint mask = group_sets_ID_range ^ p;
        for (uint p2 = mask; p2 > 0; p2 = (p2 - 1) & mask) {
            for (uint d2 = 0; d2 <= D - d1; d2++) {
                uint p1_cup_p2 = p1 | p2;
                uint new_d = max(d1, d2);
                uint merge_tree_cost = cost + host_tree[v * VAL1 + p2 * VAL2 + d2];
                uint id = v * VAL1 + p1_cup_p2 * VAL2 + new_d;
                uint old = atomicMin(&host_tree[id], merge_tree_cost);
                if (old > merge_tree_cost) {
                    uint t = atomicCAS(&updated[id], 0, 1);
                    if (!t) {
                        uint now = atomicAdd(queue_size2, 1);
                        queue2[now].v = v;
                        queue2[now].p = p1_cup_p2;
                        queue2[now].d = new_d;
                    }
                }
            }
        }
    }
}

__global__ void count_set(uint *tree,uint val1,uint val2,uint width,uint inf, uint N,int D,uint *counts)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		uint vline = idx * val1;
		for (uint x = 1; x < width; x++)
		{
            for (size_t j = 0; j <= D; j++)
            {
          	if(tree[vline+x*val2+j]!=inf)
			{
			atomicAdd(counts,1);
			}		
            }
            
			
		}
	}
}
graph_hash_of_mixed_weighted DP_gpu(CSR_graph &graph, std::vector<uint> &cumpulsory_group_vertices, graph_v_of_v_idealID &group_graph, graph_v_of_v_idealID &input_graph, int D,double *rt,int &real_cost,long long int &RAM,records &ret, int res_weight)

{
    cudaSetDevice(3);
    int *tree_weight;
    cudaMallocManaged((void **)&tree_weight, sizeof(int));
	*tree_weight = res_weight;
    
    uint N = input_graph.size();
    uint G = cumpulsory_group_vertices.size();
    uint group_sets_ID_range = (1 << G) - 1;
    uint width  = 1<<G;
    uint V = N * (1 << G) * (D + 2);
    uint VAL1 = (1 << G) * (D + 1);
    uint VAL2 = (D + 1);
    
    uint *host_tree; // node host_tree[N][1 << G][D + 3];
    queue_element_d *queue, *queue2;
    uint *queue_size, *queue_size2, *updated,max_queue_size=0;

    int *edge = graph.all_edge;
    int *edge_weight = graph.all_edge_weight;
    int *pointer = graph.all_pointer;
    
    cudaMallocManaged((void **)&host_tree, sizeof(uint) * V);
    cudaMallocManaged((void**)&queue, V * sizeof(queue_element_d));
    cudaMallocManaged((void**)&queue_size, sizeof(int));
    cudaMallocManaged((void**)&queue2, V * sizeof(queue_element_d));
    cudaMallocManaged((void**)&queue_size2, sizeof(uint));
    cudaMallocManaged((void**)&updated, sizeof(uint) * V);

    for (uint v = 0; v < N; v++) {
        for (uint p = 0; p <= group_sets_ID_range; p++) {
            for (uint d = 0; d <= D; d++) {
                host_tree[v * VAL1 + p * VAL2 + d] = inf;
            }
        }
    }
    
    *queue_size = 0;
    *queue_size2 = 0;
    std::unordered_set<uint> contain_group_vertices;
	set_max_ID(group_graph, cumpulsory_group_vertices, host_tree, contain_group_vertices,VAL1,VAL2);
    
    for(uint v = 0; v < N; v++){
        uint group_set_ID_v =get_max(v, host_tree, width,VAL1,VAL2);
        for(uint i = 1; i <= group_set_ID_v; i <<= 1){
            if(i & group_set_ID_v){
                uint id = v * VAL1 + i * VAL2;
                host_tree[id]= 0;
              
                queue[*queue_size] = queue_element_d(v, i, 0);
                (*queue_size)++;
            }
        }
    }
    
    int threadsPerBlock = 1024;
    int numBlocks = 0;
    
    auto pbegin = std::chrono::high_resolution_clock::now();
    long long int tot_process = 0;
    int rounds = 0,first_set=0;
    uint *counts;
    cudaMallocManaged((void **)&counts,sizeof(uint));
    while (*queue_size > 0) {
       // cout<<"rounds "<<rounds++<<endl;
        cudaMemset(updated, 0, V * sizeof(uint));
        numBlocks = (*queue_size + threadsPerBlock - 1) / threadsPerBlock;
        tot_process+=*queue_size;
        Relax <<< numBlocks, threadsPerBlock >>> (
            pointer, edge, edge_weight,
            queue, queue_size,
            queue2, queue_size2,
            host_tree, updated, 
            VAL1, VAL2, group_sets_ID_range, D,tree_weight
                                                    );
        cudaDeviceSynchronize();

        //the function get the prefix sum of updated
        //?
        if (*tree_weight == -1&&first_set==0)
		{
			first_set=1;
			auto mid_end = std::chrono::high_resolution_clock::now();
			ret.mid_time = std::chrono::duration_cast<std::chrono::nanoseconds>(mid_end - pbegin).count() / 1e9;
			count_set<<<(V + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(host_tree,VAL1,VAL2,width,inf,N,D,counts);
			cudaDeviceSynchronize();
			ret.mid_counts = *counts;
			ret.mid_process_queue_num = tot_process;
			//cout<<"mid_counts "<<ret.mid_counts<<" mid_process_queue_num "<<ret.mid_process_queue_num<<endl;
		}
        swap(queue, queue2);
        swap(queue_size, queue_size2);
        max_queue_size = max(max_queue_size,*queue_size);
        *queue_size2 = 0;
    }
    
  auto pend = std::chrono::high_resolution_clock::now();
	double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(pend - pbegin).count() / 1e9; // s
	*rt = runningtime;
//	std ::cout << "gpu cost time " << runningtime << std ::endl;
    std :: queue<queue_element_d> Q;
    graph_hash_of_mixed_weighted solution_tree;
    uint ans = inf;
    queue_element_d pos;
    for (uint i = 0; i < N; i++) {
        for (uint d = 0; d <= D; d++) {
            uint now = host_tree[VAL1 * i + VAL2 * group_sets_ID_range + d];
            if (ans > now) {
                ans = now;
                pos = queue_element_d(i, group_sets_ID_range, d);
            }
        }
    }
       
   
    real_cost = ans;
  
    Q.push(pos);
 
   	
	*counts = 0;
	count_set<<<(V + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(host_tree,VAL1,VAL2,width,inf,N,D,counts);
    cudaDeviceSynchronize();
    ret.counts = *counts;
    ret.process_queue_num = tot_process;
    RAM = (*counts);
    
	RAM += max_queue_size+N*width*D;
    //cout<<*counts<<" "<<max_queue_size<<" "<<N*width*D<<" "<<RAM<<endl;
    cudaFree(host_tree);
    cudaFree(queue);
    cudaFree(queue2);
    cudaFree(updated);  
    cudaFree(queue_size);
    cudaFree(queue_size2);
    return solution_tree;
}
