#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

void prefix_sum(__local int* xs) {
    const int grp = get_local_size(0);
    const int i = get_local_id(0);
    const int a = xs[i];

    for (int bit = 0; (1 << bit) < grp; bit++) {
        const int step = 1 << bit;
        const int pos = (i / step) * step;
        if (i % (step * 2) < step) {
            xs[i] += xs[pos + step];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int b = xs[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    xs[i] = b + a - xs[i];
//    printf("%d\n", xs[i]);
}

__kernel void radix(__global unsigned int* as,
                    __global unsigned int* bs,
                    __global unsigned int* cs,
                    __local int* x0s,
                    __local int* x1s,
                    __local int* xx,
                    int bit)
{
    const int gi = get_global_id(0);
    const int i = get_local_id(0);
    xx[i] = as[gi];

    const int is1 = (xx[i] >> bit) & 1;
    x1s[i] = is1;

    const int is0 = 1 - is1;
    x0s[i] = is0;

    barrier(CLK_LOCAL_MEM_FENCE);

    prefix_sum(x0s);
    prefix_sum(x1s);

    barrier(CLK_LOCAL_MEM_FENCE);
//    printf("# %d %d %d\n", i, x0s[i], x1s[i]);

    const int sz = get_local_size(0);
    const int j = is0 * x0s[i] + is1 * (x0s[sz - 1] + x1s[i]) - 1;
//    printf("%d %d %d\n", x0s[i], x0s[sz - 1], x1s[i]);

    const int grp = get_group_id(0);
    const int gj = grp * sz + j;
//    printf("%d %d %d %d\n", gj, j, i, gi);
    bs[gj] = xx[i];

    cs[grp] = x0s[sz - 1];
}

__kernel void relocate(__global unsigned int* as,
                       __global unsigned int* bs,
                       __global unsigned int* cs,
                       __local  unsigned int* ls,
                       int bit) {
    const int gi = get_global_id(0);
    const int i = get_local_id(0);
    ls[i] = as[gi];

    int zeros = 0;
    for (int j = 0; j < get_num_groups(0); j++) {
        zeros += bs[j];
    }

//    printf("%d\n", zeros);

    int off0 = 0;
    for (int j = 0; j < get_group_id(0); j++) {
        off0 += bs[j];
    }
    const int grp = get_group_id(0);
    const int off1 = get_local_size(0) * grp - off0;

    const int is1 = (ls[i] >> bit) & 1;
    const int is0 = 1 - is1;

    const int idx = is0 * (off0 + i) + is1 * (zeros + off1 + i - bs[get_group_id(0)]);
    cs[idx] = ls[i];
//    printf("%d %d\n", idx, i);
}
