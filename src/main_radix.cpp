#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


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
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 8 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    unsigned int workGroupSize = 128;
    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    std::vector<unsigned int> ps(n / workGroupSize);

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);

    gpu::gpu_mem_32u cs_gpu;
    cs_gpu.resizeN(n / workGroupSize);

    const size_t localSize = sizeof(int) * workGroupSize;
    ocl::LocalMem x0s(localSize);
    ocl::LocalMem x1s(localSize);
    ocl::LocalMem xx(localSize);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        ocl::Kernel relocate(radix_kernel, radix_kernel_length, "relocate");
        ocl::Kernel psum(radix_kernel, radix_kernel_length, "psum");
        ocl::Kernel psum_merge(radix_kernel, radix_kernel_length, "psum_merge");
        radix.compile();
        relocate.compile();
        psum.compile();
        psum_merge.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            for (int bit = 0; bit < 32; bit++) {
                radix.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, bs_gpu, cs_gpu, x0s, x1s, xx, bit);

                int szz = n / workGroupSize;
//                cs_gpu.readN(as.data(), szz);
                psum.exec(gpu::WorkSize(workGroupSize, szz), cs_gpu, szz);
//                cs_gpu.readN(as.data(), szz);
                for (int tt = 0; (workGroupSize << tt) < szz; tt++) {
                    psum_merge.exec(gpu::WorkSize(workGroupSize, szz), cs_gpu, szz, tt);
//                    cs_gpu.readN(as.data(), szz);
                }

                relocate.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        bs_gpu, cs_gpu, as_gpu, xx, bit);
                as_gpu.readN(as.data(), n);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
