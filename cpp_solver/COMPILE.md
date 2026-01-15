# C++ solver build & run (Makefile)

This folder is built via `cpp_solver/Makefile`. The Makefile compiles the sources in `cpp_solver/src/` and produces an executable named `cpp_solver/objects/scheduler.o` (despite the `.o` suffix).

## Prerequisites

- A C++23-capable compiler (default: `g++`).
- Gurobi C++ libraries and headers (linked by default).
- OR-Tools C++ libraries and headers (linked by default).
- System libs used by the link line: `lapack`, `blas`, `armadillo`, plus `pthread`, `dl`, `rt`, `tbb`.
- `cpp_solver/nlohmann/` (JSON headers) is included locally.

## Configure library paths

The Makefile uses these variables:

- `GUROBI_HOME`: root of your Gurobi installation.
  - headers: `$(GUROBI_HOME)/include`
  - libs: `$(GUROBI_HOME)/lib`
- `ORTOOLS_HOME`: root of your OR-Tools C++ distribution.
  - default: `/home/matthias/or-tools_x86_64_ubuntu-24.04_cpp_v9.10.4067`
  - headers: `$(ORTOOLS_HOME)/include`
  - libs: `$(ORTOOLS_HOME)/lib`

Examples:

```bash
make -C cpp_solver release GUROBI_HOME=/path/to/gurobi ORTOOLS_HOME=/path/to/or-tools
```

## Build targets

`make -C cpp_solver <target>`

- `debug` (default when running plain `make`): adds `-g3 -Og`
- `release`: adds `-O3 -DNDEBUG -DPARALLEL -fopenmp -fopt-info-vec=vec.opt`
- `profile`: adds `-pg -O2` (for `gprof`)
- `simple`: adds `-Os`
- `clean`: deletes `cpp_solver/objects/*.o`

All build targets end up building `scheduler` (the link step).

## Output locations

- Objects: `cpp_solver/objects/*.o`
- Executable: `cpp_solver/objects/scheduler.o`
- Vectorization report (release): `cpp_solver/vec.opt`

## Run / Makefile helpers

The Makefile also provides convenience targets that run the executable with a default argument string:

- `test`: runs `cpp_solver/objects/scheduler.o $(ARGS)`
- `analyse`: runs the program, then produces `../profiling/output.pdf` via `gprof2dot.py | dot`
- `gprof`: runs the program, then writes `../profiling/analysis.txt`

Default `ARGS` in the Makefile:

```text
--input_file=../instances/tiny.json --method=solver --lookahead 5 --gurobi_threads 12 --output_file=../solutions/tiny.json
```

Override `ARGS` at invocation time:

```bash
make -C cpp_solver test ARGS="--input_file=../instances/some.json --method=solver --output_file=../solutions/some.json"
```

Or run the binary directly:

```bash
./cpp_solver/objects/scheduler.o --input_file=instances/tiny.json --method=solver --output_file=solutions/tiny.json
```
