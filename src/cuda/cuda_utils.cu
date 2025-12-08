#ifdef USE_CUDA

#    include <iostream>

#    include "cuda/cuda_utils.cuh"

namespace lem_converter {
namespace cuda {

bool CudaDevice::is_available() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        return false;
    }
    return device_count > 0;
}

int CudaDevice::get_device_count() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

bool CudaDevice::get_device_properties(int device_id, cudaDeviceProp& prop) {
    cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
    return error == cudaSuccess;
}

bool CudaDevice::has_sufficient_memory(size_t required_bytes) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        return false;
    }
    // 10%のバッファを確保
    return free_bytes > (required_bytes * 1.1);
}

void CudaDevice::print_device_info(int device_id) {
    cudaDeviceProp prop;
    if (!get_device_properties(device_id, prop)) {
        std::cerr << "デバイス " << device_id << " のプロパティ取得に失敗しました" << std::endl;
        return;
    }

    std::cout << "CUDAデバイス " << device_id << ": " << prop.name << std::endl;
    std::cout << "  計算能力: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  総メモリ: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  マルチプロセッサ数: " << prop.multiProcessorCount << std::endl;
    std::cout << "  ブロックあたり最大スレッド数: " << prop.maxThreadsPerBlock << std::endl;
}

bool CudaDevice::set_device(int device_id) {
    cudaError_t error = cudaSetDevice(device_id);
    return error == cudaSuccess;
}

}  // namespace cuda
}  // namespace lem_converter

#endif  // USE_CUDA
