# 无共享内存版本的前缀扫描实现

## 概述

本实现提供了原始`_grid_scan_agg_2`函数的不使用共享内存(shared memory)的版本。主要修改包括：

1. 将共享内存数组替换为全局内存缓冲区
2. 使用现有的worklist作为全局内存缓冲区，避免额外的内存分配
3. 添加了block_id参数来计算正确的内存偏移
4. 保持了原有的算法逻辑和功能

## 主要修改

### 1. 新的_device_函数

**文件**: `include/prefix_scan.cuh`

- **函数名**: `_grid_scan_agg_2_no_sm`
- **主要变化**:
  - 移除了`__shared__`声明
  - 添加了全局内存缓冲区参数：`global_sml_buffer`, `global_mid_buffer`, `global_lrg_buffer`
  - 添加了`block_id`参数来计算内存偏移
  - 使用`block_offset = block_id * (warp_count_inblk + 1)`来计算每个block的偏移

### 2. 修改的reducer函数

**文件**: `include/reducer.cuh`

- **函数名**: `_push_coalesced_scan_random_list_best_atomic_2`
- **主要变化**:
  - 添加了`block_id`参数
  - 调用`_grid_scan_agg_2_no_sm`而不是`_grid_scan_agg_2`
  - 使用现有的`worklist_sml`, `worklist_mid`, `worklist_lrg`作为全局内存缓冲区

### 3. 新的kernel函数

**文件**: `include/reducer_enactor.cuh`

- **函数名**: `gen_push_worklist_no_sm`
- **主要变化**:
  - 传递`block_id`给reducer函数
  - 调用修改后的`_push_coalesced_scan_random_list_best_atomic_2`函数

### 4. 新的用户接口函数

**文件**: `include/reducer_enactor.cuh`

- **函数名**: `reducer_push_no_sm`
- **主要变化**:
  - 调用`gen_push_worklist_no_sm`而不是`gen_push_worklist`

## 使用方法

### 方法1: 使用高级接口

```cpp
// 直接调用无共享内存版本，无需额外内存分配
int result = reducer_push_no_sm(level, ggraph, mdata, worklist_gather);
```

### 方法2: 直接使用_device_函数

```cpp
// 在kernel中直接调用
_grid_scan_agg_2_no_sm<data_t, index_t>(
    thd_id_inwarp, warp_id_inblk, warp_count_inblk,
    input_sml, input_mid, input_lrg,
    output_sml, output_mid, output_lrg,
    total_sz_sml, total_sz_mid, total_sz_lrg,
    worklist_sml, worklist_mid, worklist_lrg,  // 使用现有的worklist
    block_id
);
```

## 内存使用

**优势**: 无需额外的内存分配
- 直接使用现有的`worklist_sml`, `worklist_mid`, `worklist_lrg`作为缓冲区
- 每个block使用独立的内存区域，避免冲突
- 内存偏移计算：`block_offset = block_id * (warp_count_inblk + 1)`

**注意事项**: 
- 确保worklist有足够的空间容纳前缀扫描的临时数据
- 每个block需要`(warp_count_inblk + 1)`个元素用于前缀扫描
- 前缀扫描完成后，worklist会被正常的工作队列数据覆盖

## 性能考虑

1. **内存访问模式**: 全局内存访问比共享内存慢，但避免了共享内存的bank冲突
2. **内存带宽**: 增加了全局内存带宽使用
3. **同步**: 仍然需要`__syncthreads()`来确保正确的执行顺序
4. **内存效率**: 无需额外内存分配，复用现有worklist空间
5. **适用场景**: 当共享内存不足或需要避免bank冲突时使用

## 注意事项

1. 确保worklist有足够的空间用于前缀扫描的临时存储
2. 每个block的缓冲区大小必须足够容纳`warp_count_inblk + 1`个元素
3. 前缀扫描的临时数据会在函数执行过程中被覆盖
4. 原始版本和新版本可以并存，不会相互影响

## 文件列表

修改的文件：
- `include/prefix_scan.cuh` - 添加了`_grid_scan_agg_2_no_sm`函数
- `include/reducer.cuh` - 修改了`_push_coalesced_scan_random_list_best_atomic_2`函数
- `include/reducer_enactor.cuh` - 添加了`gen_push_worklist_no_sm`和`reducer_push_no_sm`函数

新增的文件：
- `README_NO_SHARED_MEMORY.md` - 本说明文档 