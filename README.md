# parquet_reader

C-wrapper runt Apache Arrow/Parquet för att läsa Parquet-filer som float32.

## Bygg delat bibliotek

```bash
cmake -S . -B build
cmake --build build
```

## Kör tester

```bash
ctest --test-dir build
```
