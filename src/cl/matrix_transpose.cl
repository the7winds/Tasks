#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void matrix_transpose(__global float* mat, __global float* tmat, int m, int k, __local float* loc)
{
    const int row = get_global_id(1);
    const int col = get_global_id(0);

    const int width = get_global_size(0);
    const int height = get_global_size(1);

    const float v = mat[row * width + col];

    const int lrow = get_local_id(1);
    const int lcol = get_local_id(0);
    const int lwidth = get_local_size(0);


    if (row < m && col < k) {
        loc[lrow * lwidth + lcol] = v;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (row < m && col < k) {
        const int trow = get_group_id(0) * lwidth + lrow;
        const int tcol = get_group_id(1) * lwidth + lcol;
        tmat[trow * width + tcol] = loc[lcol * lwidth + lrow];
    }
}