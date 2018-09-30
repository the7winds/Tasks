#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <cassert>
#include "cl/max_prefix_sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 5; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> numbers(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            numbers[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += numbers[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += numbers[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            gpu::gpu_mem_32i nums, maxs, idxs;
            const unsigned workGroupSize = 32;
            const unsigned N = (n / workGroupSize + (n % workGroupSize ? 1 : 0)) * workGroupSize;

            std::vector<int> as(N);
            std::vector<int> bs(N);
            std::vector<int> cs(N);

            nums.resizeN(N);
            maxs.resizeN(N);
            idxs.resizeN(N);

            ocl::Kernel kernel(max_prefix_sum_kernel,  max_prefix_sum_kernel_length, "max_prefix_sum");
            kernel.compile(true);

            ocl::LocalMem cnums(workGroupSize * sizeof(int));
            ocl::LocalMem cmaxs(workGroupSize * sizeof(int));
            ocl::LocalMem cidxs(workGroupSize * sizeof(int));

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                for (int i = 0; i < N; i++) {
                    as[i] = numbers[i];
                    bs[i] = numbers[i];
                    cs[i] = i + 1;
                }

                unsigned nn = n;
                unsigned ww = 1;
                while (nn > 1) {
                    nums.writeN(as.data(), nn);
                    maxs.writeN(bs.data(), nn);
                    idxs.writeN(cs.data(), nn);

                    unsigned workSize = (nn / workGroupSize + (nn % workGroupSize ? 1 : 0)) * workGroupSize;
                    kernel.exec(gpu::WorkSize(workGroupSize, workSize), ww, nn, nums, maxs, idxs, cnums, cmaxs, cidxs);
                    nums.readN(as.data(), nn);
                    maxs.readN(bs.data(), nn);
                    idxs.readN(cs.data(), nn);

                    unsigned i = 0;
                    while (i * workGroupSize < nn) {
                        as[i] = as[i * workGroupSize];
                        bs[i] = bs[i * workGroupSize];
                        cs[i] = cs[i * workGroupSize];
                        i++;
                    }
                    nn = i;
                }

                int mx = 0;
                int idx = 0;
                maxs.readN(&mx, 1);
                idxs.readN(&idx, 1);
                mx = std::max(mx, 0);
                idx = mx ? idx : 0;

                EXPECT_THE_SAME(reference_max_sum,mx, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, idx, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
