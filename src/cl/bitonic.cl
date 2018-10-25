#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void bitonic(__global float* as, int n, int k, int p) {
    int i = get_global_id(0);
    if (i >= n) return;

    int seg = i / k;
    int seg_i = i % k;
    int seg_begin = seg * k;
    int seg_end = (seg + 1) * k;
    if (seg_i >= k / 2) return;
    int j = -1;
    if (p == 1) {
        j = seg_begin + k - seg_i - 1;
    } else {
        j = i + k / 2;
    }

//    printf("%d %d %d\n", k, i, j);

    if (as[i] > as[j]) {
        float t = as[j];
        as[j] = as[i];
        as[i] = t;
    }
}
