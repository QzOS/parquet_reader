#include "parquet_f32.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::string make_temp_path() {
    auto base = std::filesystem::temp_directory_path();
    std::string name = "pq_reader_test_" + std::to_string(std::rand()) + ".parquet";
    return (base / name).string();
}

std::shared_ptr<arrow::Table> make_test_table() {
    arrow::Int64Builder time_builder;
    arrow::FloatBuilder value_builder;

    std::vector<int64_t> times = {1000000000LL, 2000000000LL, 3000000000LL};
    std::vector<float> values = {1.5f, -2.0f, 3.25f};

    if (!time_builder.AppendValues(times).ok()) {
        return nullptr;
    }
    if (!value_builder.AppendValues(values).ok()) {
        return nullptr;
    }

    std::shared_ptr<arrow::Array> time_array;
    std::shared_ptr<arrow::Array> value_array;
    if (!time_builder.Finish(&time_array).ok()) {
        return nullptr;
    }
    if (!value_builder.Finish(&value_array).ok()) {
        return nullptr;
    }

    auto schema = arrow::schema({
        arrow::field("time", arrow::int64()),
        arrow::field("value", arrow::float32())
    });

    return arrow::Table::Make(schema, {time_array, value_array});
}

bool write_parquet(const std::string& path, const std::shared_ptr<arrow::Table>& table) {
    auto maybe_file = arrow::io::FileOutputStream::Open(path);
    if (!maybe_file.ok()) {
        std::cerr << "Failed to open output file: " << maybe_file.status().ToString() << "\n";
        return false;
    }
    std::shared_ptr<arrow::io::FileOutputStream> file = *maybe_file;

    auto status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), file, 1024);
    if (!status.ok()) {
        std::cerr << "Failed to write parquet: " << status.ToString() << "\n";
        return false;
    }
    return true;
}

bool verify_reader(const std::string& path) {
    pqh_t* handle = pq_open(path.c_str());
    if (!handle) {
        std::cerr << "pq_open failed\n";
        return false;
    }

    if (pq_num_columns(handle) != 2) {
        std::cerr << "Unexpected column count\n";
        pq_close(handle);
        return false;
    }

    if (std::string(pq_column_name(handle, 0)) != "time") {
        std::cerr << "Unexpected column name for time\n";
        pq_close(handle);
        return false;
    }

    if (pq_column_type(handle, 1) != PQ_T_F32) {
        std::cerr << "Unexpected column type for value\n";
        pq_close(handle);
        return false;
    }

    const int cols[] = {1};
    const float** out_cols = nullptr;
    const double* out_x = nullptr;
    int32_t out_nrows = 0;

    int rc = pq_read_rows_f32(handle, 0, 3, cols, 1, &out_cols, &out_x, &out_nrows);
    if (rc <= 0 || out_nrows != 3 || !out_cols || !out_x) {
        std::cerr << "pq_read_rows_f32 failed\n";
        pq_close(handle);
        return false;
    }

    const float expected_vals[] = {1.5f, -2.0f, 3.25f};
    const double expected_times[] = {1.0, 2.0, 3.0};

    for (int i = 0; i < out_nrows; ++i) {
        if (out_cols[0][i] != expected_vals[i]) {
            std::cerr << "Value mismatch at " << i << "\n";
            pq_close(handle);
            return false;
        }
        if (out_x[i] != expected_times[i]) {
            std::cerr << "Time mismatch at " << i << "\n";
            pq_close(handle);
            return false;
        }
    }

    pq_close(handle);
    return true;
}
}

int main() {
    auto table = make_test_table();
    if (!table) {
        std::cerr << "Failed to build test table\n";
        return 1;
    }

    std::string path = make_temp_path();
    if (!write_parquet(path, table)) {
        return 1;
    }

    bool ok = verify_reader(path);
    std::error_code ec;
    std::filesystem::remove(path, ec);

    return ok ? 0 : 1;
}
