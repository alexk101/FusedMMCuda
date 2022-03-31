cmake_minimum_required(VERSION 3.21)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.6/bin/nvcc")

project(FusedCuda CUDA)
add_executable(FusedCuda sddmm_spmm_csr.cu)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "Could not find CUDA toolkit")
endif()
if (NOT DEFINED $ENV{CUDA_HOME})
    set($ENV{CUDA_HOME} ${CUDAToolkit_LIBRARY_ROOT})
endif ()

set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")

target_link_libraries( FusedCuda ${CUDA_cusparse_LIBRARY} )
#message(FATAL_ERROR "Cuda Runtime exists at ${CUDA_cusparse_LIBRARY}")

include_directories(${CUDA_TOOLKIT_INCLUDE_DIRECTORIES} common)
target_compile_options(FusedCuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

set(
  CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -lcublas_static -lcublasLt_static -lcusparse_static -lculibos -lineinfo -gencode arch=compute_75,code=sm_75 --ptxas-options=-v --use_fast_math
)

set_target_properties(
  FusedCuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)


