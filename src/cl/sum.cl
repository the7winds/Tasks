#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define WORK_GROUP_SIZE 32

__kernel void sumall(__global int* numbers, int n, __local int* cache, __global int* res) {
    int i = 2 * get_global_id(0);
    int li = get_local_id(0);
    int ls = get_local_size(0);

    cache[li] = (i > n ? 0 : 1) * numbers[i] + (i + 1 > n ? 0 : 1) * numbers[i + 1];

    #if (WARP_SIZE < WORK_GROUP_SIZE)
        barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    for (int cd = 1; cd < ls; cd *= 2) {
        int cc = (cd - 1) | 1;
        if ((li & cc) == 0 && li + cd < ls) {
            cache[li] += cache[li + cd];
        }

        #if (WARP_SIZE < WORK_GROUP_SIZE)
            barrier(CLK_LOCAL_MEM_FENCE);
        #endif
    }

    if (li == 0) {
        atom_add(res, cache[0]);
    }
}