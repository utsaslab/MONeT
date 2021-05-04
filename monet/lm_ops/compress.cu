#include<cstdio>
#include<cstdlib>
#include<cstdint>
#include<cusparse.h>

#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
                             fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
                             exit(-1);}} while(0)
#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)

// void get_nz_cuda(const float* ip, int *nnzPerRow, int nz_ptr, int lda) {
//     cusparseHandle_t  handle;
//     CUSPARSE_CALL( cusparseCreate(&handle) );
//     cusparseMatDescr_t descrX;
//     CUSPARSE_CALL(cusparseCreateMatDescr(&descrX));
//     CUSPARSE_CALL( cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, lda, 256, descrX, ip,
//                                 lda, nnzPerRow, &nz_ptr));
// }

void compress_csr_256_gpu(const float* ip, float* cip, int* idx, int * rowidx, const int * nnzPerRow, size_t N) {
    cusparseHandle_t  handle;
    CUSPARSE_CALL( cusparseCreate(&handle) );
    cusparseMatDescr_t descrX;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrX));
    CUSPARSE_CALL( cusparseSdense2csr( handle, (N+255)/256, 256, descrX, ip,
                                    (N+255)/256, nnzPerRow, cip,
                                   rowidx, idx )) ;
}

void uncompress_csr_256_gpu(const float* compIP, const int * csrIdx, const int* csrRowIdx, float* op, size_t N){
    cusparseHandle_t  handle;
    CUSPARSE_CALL( cusparseCreate(&handle) );
    cusparseMatDescr_t descrX;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrX));
    CUSPARSE_CALL( cusparseScsr2dense( handle, (N+255)/256, 256, descrX, compIP,
                                   csrRowIdx, csrIdx,
                                   op,(N+255)/256 )) ;
}
