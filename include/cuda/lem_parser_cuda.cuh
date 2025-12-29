#pragma once

#ifdef USE_CUDA

#    include <cuda_runtime.h>

#    include <string>
#    include <vector>

namespace lem_converter {
namespace cuda {

// CUDA高速化LEMパース
class LemParserCuda {
   public:
    // 出力は行優先順序の1次元ベクタ (y * nx + x)
    static bool parse(const char* file_data, size_t file_size, int nx, int ny, float nodata_value,
                      std::vector<float>& output);

    // データサイズに基づいてこの操作にCUDAを使用すべきかチェック
    static bool should_use_cuda(int nx, int ny);

    // CUDAを使用するための最小データサイズ閾値を取得
    static constexpr size_t MIN_ELEMENTS_FOR_CUDA = 1000 * 1000;  // 100万要素
};

}  // namespace cuda
}  // namespace lem_converter

#endif  // USE_CUDA
