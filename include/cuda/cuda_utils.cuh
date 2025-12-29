#pragma once

#ifdef USE_CUDA

#    include <cuda_runtime.h>

#    include <iostream>
#    include <string>

namespace lem_converter {
namespace cuda {

// CUDAエラーチェックマクロ
#    define CUDA_CHECK(call)                                                       \
        do {                                                                       \
            cudaError_t error = call;                                              \
            if (error != cudaSuccess) {                                            \
                std::cerr << "CUDAエラー " << __FILE__ << ":" << __LINE__ << " - " \
                          << cudaGetErrorString(error) << std::endl;               \
                return false;                                                      \
            }                                                                      \
        } while (0)

#    define CUDA_CHECK_LAST()                                                                  \
        do {                                                                                   \
            cudaError_t error = cudaGetLastError();                                            \
            if (error != cudaSuccess) {                                                        \
                std::cerr << "CUDAカーネルエラー: " << cudaGetErrorString(error) << std::endl; \
                return false;                                                                  \
            }                                                                                  \
        } while (0)

class CudaDevice {
   public:
    // ランタイムでCUDAが利用可能かチェック
    static bool is_available();

    // CUDAデバイス数を取得
    static int get_device_count();

    // デバイスプロパティを取得
    static bool get_device_properties(int device_id, cudaDeviceProp& prop);

    // デバイスが操作に十分なメモリを持っているかチェック
    static bool has_sufficient_memory(size_t required_bytes);

    // デバイス情報を表示
    static void print_device_info(int device_id = 0);

    // デバイスを設定
    static bool set_device(int device_id = 0);
};

template <typename T>
class DeviceBuffer {
   public:
    DeviceBuffer() : ptr_(nullptr), size_(0) {}

    explicit DeviceBuffer(size_t size) : ptr_(nullptr), size_(0) { allocate(size); }

    ~DeviceBuffer() { free(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    bool allocate(size_t size) {
        free();
        cudaError_t error = cudaMalloc(&ptr_, size * sizeof(T));
        if (error != cudaSuccess) {
            std::cerr << "デバイスメモリの確保に失敗しました: " << cudaGetErrorString(error)
                      << std::endl;
            return false;
        }
        size_ = size;
        return true;
    }

    void free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    bool copy_from_host(const T* host_ptr, size_t count) {
        if (count > size_)
            return false;
        cudaError_t error = cudaMemcpy(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice);
        return error == cudaSuccess;
    }

    bool copy_to_host(T* host_ptr, size_t count) {
        if (count > size_)
            return false;
        cudaError_t error = cudaMemcpy(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
        return error == cudaSuccess;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

   private:
    T* ptr_;
    size_t size_;
};

}  // namespace cuda
}  // namespace lem_converter

#endif  // USE_CUDA
