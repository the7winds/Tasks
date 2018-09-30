#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 32

__kernel void max_prefix_sum(__global int* nums, __global int* mins, __global int* idxs, int iter,
                             __local int* cnums, __local int* cmins, __local int* cidxs) {
    int i = get_global_id(0);
    int li = get_local_id(0);
    int ls = get_local_size(0);

    cnums[li] = nums[i];
    cmins[li] = mins[i];
    cidxs[li] = cidxs[i];

#if (WARP_SIZE < WORK_GROUP_SIZE) {
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    for (int c = 1; c < ls; c *= 2) {
        if ((li & c) == 0) {
            cidxs[li] = (mins[li] < cnums[li] + mins[li + c] ? li * (1 << iter) : idxs[li + c]);
            cidxs[li] = min(mins[li], cnums[li] + mins[li + c]);
            cnums[li] += cnums[li + c];
        }

#if (WARP_SIZE < WORK_GROUP_SIZE)
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }

    int gi = get_group_id(0);
    nums[gi] = cnums[0];
    mins[gi] = cmins[0];
    idxs[gi] = cidxs[0];
}
