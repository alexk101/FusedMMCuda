// System includes
#include <iostream>
// #include <cassert>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// Sparse Operations, cusparseSpMM and cusparseSDDMM
#include <cusparse.h>

using namespace std;

// Class for Matrix Multiplication Components
class MatMul {
public:
    int A_num_rows, A_num_cols, B_num_rows, B_num_cols, C_nnz, lda, ldb, A_size, B_size, *hC_offsets, *hC_columns;
    float *matA, *matB, *hC_values, *hC_result, alpha, beta;

    MatMul(float *matA_in, int A_num_rows_in, int A_num_cols_in, int A_size_in, int lda_in, float *matB_in,
           int B_num_rows_in, int B_num_cols_in, int B_size_in, int ldb_in, int C_nnz_in, int *hC_offsets_in,
           int *hC_columns_in, float *hC_values_in, float *hC_result_in, float alpha_in, float beta_in) {

        if (A_num_cols_in != B_num_rows_in) {
            throw std::invalid_argument("Number of columns in matrix A does not equal number of rows in matrix B.");
        }
        matA = matA_in;
        A_num_rows = A_num_rows_in;
        A_num_cols = A_num_cols_in;
        A_size = A_size_in;
        lda = lda_in;

        matB = matB_in;
        B_num_rows = B_num_rows_in;
        B_num_cols = B_num_cols_in;
        B_size = B_size_in;
        ldb = ldb_in;

        C_nnz = C_nnz_in;
        hC_offsets = hC_offsets_in;
        hC_columns = hC_columns_in;
        hC_values = hC_values_in;
        hC_result = hC_result_in;

        alpha = alpha_in;
        beta = beta_in;
    }
};

void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

void calcPerf(float msecTotal, int nIter, int A_rows, int A_cols, int B_rows) {
    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(A_rows) *
                               static_cast<double>(A_cols) *
                               static_cast<double>(B_rows);
    double gigaFlops =
            (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
           gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);
}

bool validate(int resultRows, int resultCols, float *h_C, int A_rows) {
    float valB = 0.01f;
    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6; // machine zero

    for (int i = 0; i < static_cast<int>(resultRows * resultCols); i++) {
        double abs_err = fabs(h_C[i] - (A_rows * valB));
        double dot_length = A_rows;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
                   h_C[i], A_rows * valB, eps);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
    return correct;
}

// Kernel
bool _sddmm(MatMul input, cudaStream_t stream) {
    //--------------------------------------------------------------------------
    // Device memory management
    int *dC_offsets, *dC_columns;
    float *dC_values, *dB, *dA;
    checkCudaErrors(cudaMalloc((void **) &dA, input.A_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &dB, input.B_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &dC_offsets,
                               (input.A_num_rows + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &dC_columns, input.C_nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &dC_values, input.C_nnz * sizeof(float)));

    checkCudaErrors(cudaMemcpy(dA, input.matA, input.A_size * sizeof(float),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dB, input.matB, input.B_size * sizeof(float),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dC_offsets, input.hC_offsets,
                               (input.A_num_rows + 1) * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dC_columns, input.hC_columns, input.C_nnz * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dC_values, input.hC_values, input.C_nnz * sizeof(float),
                               cudaMemcpyHostToDevice));
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    checkCudaErrors(cusparseCreate(&handle));
    // Create dense matrix A
    checkCudaErrors(cusparseCreateDnMat(&matA, input.A_num_rows, input.A_num_cols, input.lda, dA,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Create dense matrix B
    checkCudaErrors(cusparseCreateDnMat(&matB, input.A_num_cols, input.B_num_cols, input.ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Create sparse matrix C in CSR format
    checkCudaErrors(cusparseCreateCsr(&matC, input.A_num_rows, input.B_num_cols, input.C_nnz,
                                      dC_offsets, dC_columns, dC_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // allocate an external buffer if needed
    checkCudaErrors(cusparseSDDMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &input.alpha, matA, matB, &input.beta, matC, CUDA_R_32F,
            CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize));
    checkCudaErrors(cudaMalloc(&dBuffer, bufferSize));

    // execute preprocess (optional)
    checkCudaErrors(cusparseSDDMM_preprocess(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &input.alpha, matA, matB, &input.beta, matC, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer));
    // execute SpMM
    checkCudaErrors(cusparseSDDMM(
          handle,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &input.alpha, matA, matB, &input.beta, matC, CUDA_R_32F,
          CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer));
    // destroy matrix/vector descriptors
    checkCudaErrors(cusparseDestroyDnMat(matA));
    checkCudaErrors(cusparseDestroyDnMat(matB));
    checkCudaErrors(cusparseDestroySpMat(matC));
    checkCudaErrors(cusparseDestroy(handle));
    //--------------------------------------------------------------------------
    // device result check
    // Copy result from device to host
    checkCudaErrors(
            cudaMemcpyAsync(input.hC_values, dC_values, input.C_nnz * sizeof(float),
                            cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    // B rows = C rows && A || B cols = C cols
    bool correct = validate(input.B_num_rows, input.B_num_cols, input.hC_values, input.A_num_rows);
    //--------------------------------------------------------------------------
    // device memory deallocation
    checkCudaErrors(cudaFree(dBuffer));
    checkCudaErrors(cudaFree(dA));
    checkCudaErrors(cudaFree(dB));
    checkCudaErrors(cudaFree(dC_offsets));
    checkCudaErrors(cudaFree(dC_columns));
    checkCudaErrors(cudaFree(dC_values));
    return correct;
}

void _spmm(MatMul input, cudaStream_t stream) {

}

bool _sddmmSpmm(MatMul input, cudaStream_t stream) {
    bool correct = _sddmm(input, stream);
    if(!correct) {
        printf("sddmm failed");
    }
    return correct;
}

// Main
int main() {
    // Testing Variables -> from sddmm_csr_example.c
    int A_num_rows = 4;
    int A_num_cols = 4;
    int B_num_rows = A_num_cols;
    int B_num_cols = 3;
    int C_nnz = 9;
    int lda = A_num_cols;
    int ldb = B_num_cols;
    int A_size = lda * A_num_rows;
    int B_size = ldb * B_num_rows;
    float hA[] = {1.0f, 2.0f, 3.0f, 4.0f,
                  5.0f, 6.0f, 7.0f, 8.0f,
                  9.0f, 10.0f, 11.0f, 12.0f,
                  13.0f, 14.0f, 15.0f, 16.0f};
    float hB[] = {1.0f, 2.0f, 3.0f,
                  4.0f, 5.0f, 6.0f,
                  7.0f, 8.0f, 9.0f,
                  10.0f, 11.0f, 12.0f};
    int hC_offsets[] = {0, 3, 4, 7, 9};
    int hC_columns[] = {0, 1, 2, 1, 0, 1, 2, 0, 2};
    float hC_values[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f};
    float hC_result[] = {70.0f, 80.0f, 90.0f,184.0f,246.0f,
                         288.0f, 330.0f,334.0f, 450.0f};
    float alpha = 1.0f;
    float beta = 0.0f;

    MatMul test = MatMul(
        hA,
        A_num_rows,
        A_num_cols,
        A_size,
        lda,
        hB,
        B_num_rows,
        B_num_cols,
        B_size,
        ldb,
        C_nnz,
        hC_offsets,
        hC_columns,
        hC_values,
        hC_result,
        alpha,
        beta
    );

    printf("Computing result using CUDA Kernel...\n");

    // Initialize timing variables
    cudaStream_t stream;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Performs warmup operation using _sddmmSpmm CUDA kernel
    bool correct = _sddmmSpmm(test, stream);

    printf("done\n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel
    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
        _sddmmSpmm(test, stream);
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    calcPerf(msecTotal, nIter, test.A_num_rows, test.A_num_cols, test.B_num_rows);

    // old correct code

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}