#include "parquet_f32.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

struct pqh {
    std::string last_error;

    std::shared_ptr<arrow::Table> table;
    std::shared_ptr<arrow::Schema> schema;

    std::vector<std::string> names; // stable storage for pq_column_name()

    int64_t num_rows;
    int time_col; // -1 if none

    // Output buffers valid until next pq_read_rows_f32()
    std::vector<std::vector<float> > col_bufs; // [ncols][nrows]
    std::vector<const float*> col_ptrs;        // [ncols]
    std::vector<double> x_buf;                 // [nrows] if time_col >= 0
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
    h->table.reset();
    h->schema.reset();
    h->names.clear();
    h->num_rows = 0;
    h->time_col = -1;

    // Open file
    auto maybe_file = arrow::io::ReadableFile::Open(parquet_path);
    if (!maybe_file.ok()) {
        set_err(h.get(), maybe_file.status().ToString());
        return NULL;
    }
    std::shared_ptr<arrow::io::RandomAccessFile> file = *maybe_file;

    // Build Parquet reader (version-tolerant)
    std::unique_ptr<parquet::ParquetFileReader> pq_reader;
    try {
        pq_reader = parquet::ParquetFileReader::Open(file);
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

    // Read whole table (simple)
    std::shared_ptr<arrow::Table> table;
    st = reader->ReadTable(&table);
    if (!st.ok()) {
        set_err(h.get(), st.ToString());
        return NULL;
    }
    if (!table) {
        set_err(h.get(), "ReadTable returned NULL table");
        return NULL;
    }

    // Ensure single chunk per column (keeps the rest of the code simple/correct)
    auto combined_result = table->CombineChunks(arrow::default_memory_pool());
    if (!combined_result.ok()) {
        set_err(h.get(), combined_result.status().ToString());
        return NULL;
    }
    table = *combined_result;

    h->table = table;
    h->schema = table->schema();
    h->num_rows = table->num_rows();
    h->time_col = detect_time_column(h->schema);

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

static int col_to_f32(pqh* h, int col_index, int64_t row0, int32_t nrows, std::vector<float>& out) {
    if (!h || !h->table) return -1;
    if (col_index < 0 || col_index >= h->table->num_columns()) return -2;

    auto chunked = h->table->column(col_index);
    if (!chunked || chunked->num_chunks() <= 0) return -3;

    // Simplification: first chunk only
    auto arr = chunked->chunk(0);
    if (!arr) return -4;

    switch (arr->type_id()) {
    case arrow::Type::BOOL:
        cast_numeric_to_f32<arrow::BooleanArray>(arr, row0, nrows, out,
            [](const arrow::BooleanArray* a, int64_t i) -> int { return a->Value(i) ? 1 : 0; });
        return 0;
    case arrow::Type::INT32:
        cast_numeric_to_f32<arrow::Int32Array>(arr, row0, nrows, out,
            [](const arrow::Int32Array* a, int64_t i) -> int32_t { return a->Value(i); });
        return 0;
    case arrow::Type::INT64:
        cast_numeric_to_f32<arrow::Int64Array>(arr, row0, nrows, out,
            [](const arrow::Int64Array* a, int64_t i) -> int64_t { return a->Value(i); });
        return 0;
    case arrow::Type::FLOAT:
        cast_numeric_to_f32<arrow::FloatArray>(arr, row0, nrows, out,
            [](const arrow::FloatArray* a, int64_t i) -> float { return a->Value(i); });
        return 0;
    case arrow::Type::DOUBLE:
        cast_numeric_to_f32<arrow::DoubleArray>(arr, row0, nrows, out,
            [](const arrow::DoubleArray* a, int64_t i) -> double { return a->Value(i); });
        return 0;
    default:
        return -10;
    }
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

    if (!h || !h->table) return -1;
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

    h->col_bufs.assign((size_t)ncols, std::vector<float>());
    h->col_ptrs.assign((size_t)ncols, (const float*)NULL);

    for (int32_t i = 0; i < ncols; i++) {
        h->col_bufs[(size_t)i].assign((size_t)nrows, 0.0f);

        const int col_index = cols[i];
        const int rc = col_to_f32(h, col_index, row0, nrows, h->col_bufs[(size_t)i]);
        if (rc < 0) {
            set_err(h, "Unsupported type or failed conversion at column index " + std::to_string(col_index));
            return -20;
        }
        h->col_ptrs[(size_t)i] = h->col_bufs[(size_t)i].data();
    }

    if (out_x) {
        if (h->time_col >= 0) {
            h->x_buf.assign((size_t)nrows, std::numeric_limits<double>::quiet_NaN());

            auto chunked = h->table->column(h->time_col);
            if (chunked && chunked->num_chunks() > 0) {
                auto arr = chunked->chunk(0);
                fill_x_buf(h, arr, row0, nrows);
                *out_x = h->x_buf.data();
            }
            else {
                *out_x = NULL;
            }
        }
        else {
            *out_x = NULL;
        }
    }

    *out_nrows = nrows;
    *out_cols = (const float**)h->col_ptrs.data();
    return nrows;
}

const char* pq_last_error(pqh_t* hh) {
    pqh* h = reinterpret_cast<pqh*>(hh);
    if (!h) return "pqh is NULL";
    return h->last_error.c_str();
}
