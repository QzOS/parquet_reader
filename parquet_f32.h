#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct pqh pqh_t;

    typedef enum {
        PQ_T_BOOL = 1,
        PQ_T_I32,
        PQ_T_I64,
        PQ_T_F32,
        PQ_T_F64,
        PQ_T_STR,
        PQ_T_BIN,
        PQ_T_TS,
        PQ_T_OTHER
    } pq_type_t;

    pqh_t* pq_open(const char* parquet_path);
    void pq_close(pqh_t* h);

    int pq_num_columns(pqh_t* h);
    const char* pq_column_name(pqh_t* h, int col);     // valid until close
    pq_type_t pq_column_type(pqh_t* h, int col);

    // Read rows [row0, row0+nrows_req), return:
    // - out_x: double seconds (absolute time, typically epoch seconds)
    // - out_cols[i]: float32 y-values per requested column
    //
    // Lifetime: returned pointers valid until next pq_read_rows_f32() on same handle or pq_close().
    // NULLs in numeric columns -> NaN in float32.
    // If no time column exists, out_x will be NULL.
    int pq_read_rows_f32(
        pqh_t* h,
        int64_t row0,
        int32_t nrows_req,
        const int* cols,
        int32_t ncols,
        const float*** out_cols,   // out: [ncols] of float*
        const double** out_x,      // out: double[nrows] or NULL
        int32_t* out_nrows
    );

    const char* pq_last_error(pqh_t* h);

#ifdef __cplusplus
}
#endif
