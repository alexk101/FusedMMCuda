#include <iostream>

int main() {
    cudaError_t err = cudaMalloc((void**) &dA, A_size * sizeof(float));
    CHECK_CUDA(err, __LINE__-1);
    err = cudaMalloc((void**) &dB, B_size * sizeof(float));
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
