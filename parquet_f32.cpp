#include "parquet_f32.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

struct pqh {
    std::string last_error;

    std::shared_ptr<arrow::io::RandomAccessFile> file;
    std::unique_ptr<parquet::arrow::FileReader> reader;
    std::shared_ptr<arrow::Schema> schema;

    std::vector<std::string> names; // stable storage for pq_column_name()

    int64_t num_rows;
    int time_col; // -1 if none

    // Row group layout (prefix sums): map global row -> row group.
    std::vector<int64_t> rg_row0;
    std::vector<int64_t> rg_nrows;

    // Output buffers valid until next pq_read_rows_f32()
    std::vector<std::vector<float> > col_bufs; // [ncols][nrows]
    std::vector<const float*> col_ptrs;        // [ncols]
    std::vector<double> x_buf;                 // cached X (time) or per-call

    // X cache key (valid when x_cached == true)
    bool x_cached;
    int64_t x_row0;
    int32_t x_nrows;
};

static void set_err(pqh* h, const std::string& s) { if (h) h->last_error = s; }

static pq_type_t map_arrow_type(const std::shared_ptr<arrow::DataType>& t) {
    if (!t) return PQ_T_OTHER;
    switch (t->id()) {
    case arrow::Type::BOOL: return PQ_T_BOOL;
    case arrow::Type::INT32: return PQ_T_I32;
    case arrow::Type::INT64: return PQ_T_I64;
    case arrow::Type::FLOAT: return PQ_T_F32;
    case arrow::Type::DOUBLE: return PQ_T_F64;
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING: return PQ_T_STR;
    case arrow::Type::BINARY:
    case arrow::Type::LARGE_BINARY: return PQ_T_BIN;
    case arrow::Type::TIMESTAMP: return PQ_T_TS;
    default: return PQ_T_OTHER;
    }
}

static int detect_time_column(const std::shared_ptr<arrow::Schema>& schema) {
    if (!schema) return -1;
    const int n = schema->num_fields();

    for (int i = 0; i < n; i++) {
        const std::string name = schema->field(i)->name();
        const auto t = schema->field(i)->type();
        const bool name_time =
            (name == "time") || (name == "Time") || (name == "timestamp") || (name == "Timestamp");
        if (name_time) {
            if (t->id() == arrow::Type::INT64 || t->id() == arrow::Type::TIMESTAMP) return i;
        }
    }
    for (int i = 0; i < n; i++) {
        const auto t = schema->field(i)->type();
        if (t->id() == arrow::Type::TIMESTAMP || t->id() == arrow::Type::INT64) return i;
    }
    return -1;
}

pqh_t* pq_open(const char* parquet_path) {
    if (!parquet_path) return NULL;

    std::unique_ptr<pqh> h(new pqh());
    h->last_error.clear();
    h->file.reset();
    h->reader.reset();
    h->schema.reset();
    h->names.clear();
    h->num_rows = 0;
    h->time_col = -1;
    h->rg_row0.clear();
    h->rg_nrows.clear();
    h->x_cached = false;
    h->x_row0 = 0;
    h->x_nrows = 0;

    // Open file
    auto maybe_file = arrow::io::ReadableFile::Open(parquet_path);
    if (!maybe_file.ok()) {
        set_err(h.get(), maybe_file.status().ToString());
        return NULL;
    }
    h->file = *maybe_file;

    // Build Parquet reader (version-tolerant)
    std::unique_ptr<parquet::ParquetFileReader> pq_reader;
    try {
        pq_reader = parquet::ParquetFileReader::Open(h->file);
    } catch (const std::exception& e) {
        set_err(h.get(), std::string("ParquetFileReader::Open failed: ") + e.what());
        return NULL;
    } catch (...) {
        set_err(h.get(), "ParquetFileReader::Open failed: unknown exception");
        return NULL;
    }

    std::unique_ptr<parquet::arrow::FileReader> reader;
    auto st = parquet::arrow::FileReader::Make(
        arrow::default_memory_pool(),
        std::move(pq_reader),
        &reader);

    if (!st.ok()) {
        set_err(h.get(), st.ToString());
        return NULL;
    }

    h->reader = std::move(reader);

    // Read schema only (no data)
    std::shared_ptr<arrow::Schema> schema;
    st = h->reader->GetSchema(&schema);
    if (!st.ok() || !schema) {
        set_err(h.get(), st.ToString());
        return NULL;
    }
    h->schema = schema;
    h->time_col = detect_time_column(h->schema);

    // Row group metadata (no data)
    auto md = h->reader->parquet_reader()->metadata();
    if (!md) {
        set_err(h.get(), "Parquet metadata is NULL");
        return NULL;
    }
    h->num_rows = md->num_rows();
    const int nrg = md->num_row_groups();
    h->rg_row0.resize((size_t)nrg);
    h->rg_nrows.resize((size_t)nrg);
    int64_t acc = 0;
    for (int rg = 0; rg < nrg; rg++) {
        const int64_t n = md->RowGroup(rg)->num_rows();
        h->rg_row0[(size_t)rg] = acc;
        h->rg_nrows[(size_t)rg] = n;
        acc += n;
    }

    // Cache stable column names
    h->names.resize((size_t)h->schema->num_fields());
    for (int i = 0; i < h->schema->num_fields(); i++) {
        h->names[(size_t)i] = h->schema->field(i)->name();
    }

    return h.release();
}

void pq_close(pqh_t* hh) {
    pqh* h = reinterpret_cast<pqh*>(hh);
    delete h;
}

int pq_num_columns(pqh_t* hh) {
    pqh* h = reinterpret_cast<pqh*>(hh);
    if (!h || !h->schema) return -1;
    return h->schema->num_fields();
}

const char* pq_column_name(pqh_t* hh, int col) {
    pqh* h = reinterpret_cast<pqh*>(hh);
    if (!h) return NULL;
    if (col < 0 || col >= (int)h->names.size()) return NULL;
    return h->names[(size_t)col].c_str();
}

pq_type_t pq_column_type(pqh_t* hh, int col) {
    pqh* h = reinterpret_cast<pqh*>(hh);
    if (!h || !h->schema) return PQ_T_OTHER;
    if (col < 0 || col >= h->schema->num_fields()) return PQ_T_OTHER;
    return map_arrow_type(h->schema->field(col)->type());
}

// ---- time -> double seconds ----

// Policy for INT64 time columns (non-timestamp).
// If your CSV time column is epoch nanoseconds (common), keep this.
// If it's microseconds or milliseconds, change the scale.
static double int64_time_to_seconds(int64_t t) {
    // assume nanoseconds
    return (double)t * 1e-9;
}

static double timestamp_value_to_seconds(const arrow::TimestampType& ts_type, int64_t v) {
    // v is integer timestamp in unit given by ts_type.unit()
    switch (ts_type.unit()) {
    case arrow::TimeUnit::SECOND:      return (double)v;
    case arrow::TimeUnit::MILLI:       return (double)v * 1e-3;
    case arrow::TimeUnit::MICRO:       return (double)v * 1e-6;
    case arrow::TimeUnit::NANO:        return (double)v * 1e-9;
    default:                           return (double)v;
    }
}

// NOTE: fill_x_buf() kept for reference/debug; main path uses fill_x_buf_into() via chunk-aware copy.
static void fill_x_buf(pqh* h,
    const std::shared_ptr<arrow::Array>& arr,
    int64_t row0,
    int32_t nrows) {
    if (!h || !arr) return;

    const auto id = arr->type_id();

    if (id == arrow::Type::INT64) {
        auto a = std::static_pointer_cast<arrow::Int64Array>(arr);
        for (int32_t i = 0; i < nrows; i++) {
            const int64_t idx = row0 + (int64_t)i;
            if (idx < 0 || idx >= a->length() || a->IsNull(idx)) {
                h->x_buf[(size_t)i] = std::numeric_limits<double>::quiet_NaN();
            }
            else {
                h->x_buf[(size_t)i] = int64_time_to_seconds(a->Value(idx));
            }
        }
        return;
    }

    if (id == arrow::Type::TIMESTAMP) {
        auto a = std::static_pointer_cast<arrow::TimestampArray>(arr);
        auto ts_type = std::static_pointer_cast<arrow::TimestampType>(arr->type());
        for (int32_t i = 0; i < nrows; i++) {
            const int64_t idx = row0 + (int64_t)i;
            if (idx < 0 || idx >= a->length() || a->IsNull(idx)) {
                h->x_buf[(size_t)i] = std::numeric_limits<double>::quiet_NaN();
            }
            else {
                h->x_buf[(size_t)i] = timestamp_value_to_seconds(*ts_type, a->Value(idx));
            }
        }
        return;
    }

    // unsupported time type => NaNs
    for (int32_t i = 0; i < nrows; i++) {
        h->x_buf[(size_t)i] = std::numeric_limits<double>::quiet_NaN();
    }
}

// Offset-aware x fill (copy slice from row group array into output x_buf)
static void fill_x_buf_into(pqh* h,
    const std::shared_ptr<arrow::Array>& arr,
    int64_t src_off,
    int32_t nrows,
    int64_t dst_off) {
    if (!h || !arr) return;

    const auto id = arr->type_id();

    if (id == arrow::Type::INT64) {
        auto a = std::static_pointer_cast<arrow::Int64Array>(arr);
        for (int32_t i = 0; i < nrows; i++) {
            const int64_t si = src_off + (int64_t)i;
            const int64_t di = dst_off + (int64_t)i;
            if (si < 0 || si >= a->length() || a->IsNull(si)) {
                h->x_buf[(size_t)di] = std::numeric_limits<double>::quiet_NaN();
            } else {
                h->x_buf[(size_t)di] = int64_time_to_seconds(a->Value(si));
            }
        }
        return;
    }

    if (id == arrow::Type::TIMESTAMP) {
        auto a = std::static_pointer_cast<arrow::TimestampArray>(arr);
        auto ts_type = std::static_pointer_cast<arrow::TimestampType>(arr->type());
        for (int32_t i = 0; i < nrows; i++) {
            const int64_t si = src_off + (int64_t)i;
            const int64_t di = dst_off + (int64_t)i;
            if (si < 0 || si >= a->length() || a->IsNull(si)) {
                h->x_buf[(size_t)di] = std::numeric_limits<double>::quiet_NaN();
            } else {
                h->x_buf[(size_t)di] = timestamp_value_to_seconds(*ts_type, a->Value(si));
            }
        }
        return;
    }

    for (int32_t i = 0; i < nrows; i++) {
        const int64_t di = dst_off + (int64_t)i;
        h->x_buf[(size_t)di] = std::numeric_limits<double>::quiet_NaN();
    }
}

// ---- numeric -> float32 ----

template <typename ArrowArrayT, typename GetValueFn>
static void cast_numeric_to_f32(const std::shared_ptr<arrow::Array>& arr,
    int64_t row0,
    int32_t nrows,
    std::vector<float>& out,
    GetValueFn get_value) {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    auto a = std::static_pointer_cast<ArrowArrayT>(arr);

    for (int32_t i = 0; i < nrows; i++) {
        const int64_t idx = row0 + (int64_t)i;
        if (idx < 0 || idx >= a->length() || a->IsNull(idx)) {
            out[(size_t)i] = qnan;
        }
        else {
            out[(size_t)i] = (float)get_value(a.get(), idx);
        }
    }
}

template <typename ArrowArrayT, typename GetValueFn>
static void cast_numeric_to_f32_into(
    const std::shared_ptr<arrow::Array>& arr,
    int64_t src_off,
    int32_t nrows,
    float* dst,
    int64_t dst_off,
    GetValueFn get_value) {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    auto a = std::static_pointer_cast<ArrowArrayT>(arr);

    for (int32_t i = 0; i < nrows; i++) {
        const int64_t si = src_off + (int64_t)i;
        const int64_t di = dst_off + (int64_t)i;
        if (si < 0 || si >= a->length() || a->IsNull(si)) {
            dst[di] = qnan;
        } else {
            dst[di] = (float)get_value(a.get(), si);
        }
    }
}

static int array_to_f32_into(
    const std::shared_ptr<arrow::Array>& arr,
    int64_t src_off,
    int32_t nrows,
    float* dst,
    int64_t dst_off) {
    if (!arr || !dst) return -1;

    switch (arr->type_id()) {
    case arrow::Type::BOOL:
        cast_numeric_to_f32_into<arrow::BooleanArray>(arr, src_off, nrows, dst, dst_off,
            [](const arrow::BooleanArray* a, int64_t i) -> int { return a->Value(i) ? 1 : 0; });
        return 0;
    case arrow::Type::INT32:
        cast_numeric_to_f32_into<arrow::Int32Array>(arr, src_off, nrows, dst, dst_off,
            [](const arrow::Int32Array* a, int64_t i) -> int32_t { return a->Value(i); });
        return 0;
    case arrow::Type::INT64:
        cast_numeric_to_f32_into<arrow::Int64Array>(arr, src_off, nrows, dst, dst_off,
            [](const arrow::Int64Array* a, int64_t i) -> int64_t { return a->Value(i); });
        return 0;
    case arrow::Type::FLOAT:
        cast_numeric_to_f32_into<arrow::FloatArray>(arr, src_off, nrows, dst, dst_off,
            [](const arrow::FloatArray* a, int64_t i) -> float { return a->Value(i); });
        return 0;
    case arrow::Type::DOUBLE:
        cast_numeric_to_f32_into<arrow::DoubleArray>(arr, src_off, nrows, dst, dst_off,
            [](const arrow::DoubleArray* a, int64_t i) -> double { return a->Value(i); });
        return 0;
    default:
        return -10;
    }
}

static int find_rg_for_row(const pqh* h, int64_t row) {
    if (!h || h->rg_row0.empty()) return -1;
    int lo = 0;
    int hi = (int)h->rg_row0.size();
    while (lo + 1 < hi) {
        int mid = lo + (hi - lo) / 2;
        if (h->rg_row0[(size_t)mid] <= row) lo = mid;
        else hi = mid;
    }
    return lo;
}

// Copy numeric chunked column slice into float32 output (NaN for nulls).
static int copy_chunked_to_f32_into(
    const std::shared_ptr<arrow::ChunkedArray>& chunked,
    int64_t src_off,
    int32_t nrows,
    float* dst,
    int64_t dst_off) {

    if (!chunked || !dst) return -1;
    if (nrows <= 0) return 0;
    if (src_off < 0) return -2;

    int64_t remaining = (int64_t)nrows;
    int64_t src = src_off;
    int64_t dst_i = dst_off;

    int64_t base = 0;
    const int nch = chunked->num_chunks();
    for (int ci = 0; ci < nch && remaining > 0; ci++) {
        const std::shared_ptr<arrow::Array> arr = chunked->chunk(ci);
        if (!arr) {
            continue;
        }

        const int64_t len = arr->length();
        const int64_t chunk_start = base;
        const int64_t chunk_end = base + len; // exclusive

        if (src >= chunk_end) {
            base = chunk_end;
            continue;
        }
        if (src < chunk_start) {
            // Defensive: should not happen if base tracking is correct.
            src = chunk_start;
        }

        const int64_t in_chunk_off = src - chunk_start;
        const int64_t avail = len - in_chunk_off;
        const int64_t take64 = (remaining < avail) ? remaining : avail;
        const int32_t take = (int32_t)take64;

        const int rc = array_to_f32_into(arr, in_chunk_off, take, dst, dst_i);
        if (rc < 0) return rc;

        remaining -= take64;
        src += take64;
        dst_i += take64;

        base = chunk_end;
    }

    // If remaining > 0, the caller asked beyond available data; leave the tail as-is.
    return 0;
}

// Copy time chunked column slice into x_buf (double seconds, NaN for nulls).
static void copy_chunked_time_into(
    pqh* h,
    const std::shared_ptr<arrow::ChunkedArray>& chunked,
    int64_t src_off,
    int32_t nrows,
    int64_t dst_off) {

    if (!h || !chunked) return;
    if (nrows <= 0) return;
    if (src_off < 0) return;

    int64_t remaining = (int64_t)nrows;
    int64_t src = src_off;
    int64_t dst_i = dst_off;

    int64_t base = 0;
    const int nch = chunked->num_chunks();
    for (int ci = 0; ci < nch && remaining > 0; ci++) {
        const std::shared_ptr<arrow::Array> arr = chunked->chunk(ci);
        if (!arr) {
            continue;
        }

        const int64_t len = arr->length();
        const int64_t chunk_start = base;
        const int64_t chunk_end = base + len; // exclusive

        if (src >= chunk_end) {
            base = chunk_end;
            continue;
        }
        if (src < chunk_start) {
            src = chunk_start;
        }

        const int64_t in_chunk_off = src - chunk_start;
        const int64_t avail = len - in_chunk_off;
        const int64_t take64 = (remaining < avail) ? remaining : avail;
        const int32_t take = (int32_t)take64;

        fill_x_buf_into(h, arr, in_chunk_off, take, dst_i);

        remaining -= take64;
        src += take64;
        dst_i += take64;

        base = chunk_end;
    }
}

static arrow::Status read_rowgroup_cols(
    parquet::arrow::FileReader* r,
    int row_group,
    const std::vector<int>& col_indices,
    std::shared_ptr<arrow::Table>* out) {
    if (!r || !out) return arrow::Status::Invalid("null");
    auto rg = r->RowGroup(row_group);
    if (!rg) return arrow::Status::Invalid("RowGroup() returned null");
    return rg->ReadTable(col_indices, out);
}

int pq_read_rows_f32(
    pqh_t* hh,
    int64_t row0,
    int32_t nrows_req,
    const int* cols,
    int32_t ncols,
    const float*** out_cols,
    const double** out_x,
    int32_t* out_nrows) {

    pqh* h = reinterpret_cast<pqh*>(hh);

    if (!h || !h->reader || !h->schema) return -1;
    if (!cols || ncols <= 0) return -2;
    if (!out_cols || !out_nrows) return -3;
    if (row0 < 0) return -4;
    if (nrows_req <= 0) return -5;

    if (row0 >= h->num_rows) {
        *out_nrows = 0;
        *out_cols = NULL;
        if (out_x) *out_x = NULL;
        return 0;
    }

    int64_t remaining = h->num_rows - row0;
    int32_t nrows = nrows_req;
    if ((int64_t)nrows > remaining) nrows = (int32_t)remaining;

    h->col_bufs.assign((size_t)ncols, std::vector<float>((size_t)nrows, std::numeric_limits<float>::quiet_NaN()));
    h->col_ptrs.assign((size_t)ncols, (const float*)NULL);

    for (int32_t i = 0; i < ncols; i++) {
        h->col_ptrs[(size_t)i] = h->col_bufs[(size_t)i].data();
    }

    const bool want_x = (out_x != NULL);
    const bool have_time = (h->time_col >= 0);

    // Decide whether we need to (re)load X or can reuse cached X.
    bool need_x_load = false;
    if (want_x && have_time) {
        if (h->x_cached && h->x_row0 == row0 && h->x_nrows == nrows && (int32_t)h->x_buf.size() == nrows) {
            // cache hit
            need_x_load = false;
        } else {
            // cache miss: allocate/refresh x_buf for this interval
            need_x_load = true;
            h->x_buf.assign((size_t)nrows, std::numeric_limits<double>::quiet_NaN());
            h->x_cached = true;
            h->x_row0 = row0;
            h->x_nrows = nrows;
        }
    } else {
        // not requested or no time col -> don't promise cached x
        need_x_load = false;
        // keep x_buf as-is (could be useful for later calls); just don't return it unless applicable
    }

    const int64_t row1 = row0 + (int64_t)nrows - 1;
    const int rg_first = find_rg_for_row(h, row0);
    const int rg_last = find_rg_for_row(h, row1);
    if (rg_first < 0 || rg_last < 0) {
        set_err(h, "No row groups in file (empty parquet?)");
        return -7;
    }

    // read_cols = requested cols (+ time_col if needed)
    std::vector<int> read_cols;
    read_cols.reserve((size_t)ncols + 1);
    for (int32_t i = 0; i < ncols; i++) {
        const int c = cols[i];
        if (c < 0 || c >= h->schema->num_fields()) {
            set_err(h, "Column index out of range: " + std::to_string(c));
            return -6;
        }
        read_cols.push_back(c);
    }
    int time_pos = -1;
    if (need_x_load) {
        bool have = false;
        for (int i = 0; i < (int)read_cols.size(); i++) {
            if (read_cols[(size_t)i] == h->time_col) { have = true; time_pos = i; break; }
        }
        if (!have) { time_pos = (int)read_cols.size(); read_cols.push_back(h->time_col); }
    }

    for (int rg = rg_first; rg <= rg_last; rg++) {
        const int64_t rg0 = h->rg_row0[(size_t)rg];
        const int64_t rgN = h->rg_nrows[(size_t)rg];
        if (rgN <= 0) continue;

        // intersection of [row0,row1] and this RG
        const int64_t want0 = (row0 > rg0) ? row0 : rg0;
        const int64_t want1 = (row1 < (rg0 + rgN - 1)) ? row1 : (rg0 + rgN - 1);
        const int32_t take = (int32_t)(want1 - want0 + 1);
        const int64_t src_off = want0 - rg0;  // inside RG
        const int64_t dst_off = want0 - row0; // inside output buffers

        std::shared_ptr<arrow::Table> t;
        auto st = read_rowgroup_cols(h->reader.get(), rg, read_cols, &t);
        if (!st.ok() || !t) {
            set_err(h, std::string("RowGroup::ReadTable failed: ") + st.ToString());
            return -20;
        }

        // requested cols are first ncols columns in t (same order as read_cols)
        for (int32_t i = 0; i < ncols; i++) {
            auto chunked = t->column((int)i);
            if (!chunked) {
                set_err(h, "Missing chunked column for requested column");
                return -22;
            }
            const int rc = copy_chunked_to_f32_into(chunked, src_off, take,
                h->col_bufs[(size_t)i].data(), dst_off);
            if (rc < 0) {
                set_err(h, "Unsupported type or failed conversion at column index " + std::to_string(cols[i]));
                return -23;
            }
        }

        if (need_x_load && time_pos >= 0) {
            auto tcol = t->column(time_pos);
            if (tcol) copy_chunked_time_into(h, tcol, src_off, take, dst_off);
        }
    }

    *out_nrows = nrows;
    *out_cols = (const float**)h->col_ptrs.data();
    if (out_x) {
        if (want_x && have_time && h->x_cached && h->x_row0 == row0 && h->x_nrows == nrows && !h->x_buf.empty())
            *out_x = h->x_buf.data();
        else *out_x = NULL;
    }
    return nrows;
}

const char* pq_last_error(pqh_t* hh) {
    pqh* h = reinterpret_cast<pqh*>(hh);
    if (!h) return "pqh is NULL";
    return h->last_error.c_str();
}
