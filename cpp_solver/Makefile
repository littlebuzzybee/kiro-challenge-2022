CXX = g++
CXXFLAGS := -std=c++23 -Wall -Wextra -march=native

OPTIONAL_FLAGS := -Werror -fanalyzer -pedantic
INDIVIDUAL_FLAGS := -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function -Wno-unused-private-field -Wno-unused-value -Wno-unused-local-typedefs 

JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

ifneq (,$(filter release,$(MAKECMDGOALS)))
  ifeq (,$(filter -j%,$(MAKEFLAGS)))
    MAKEFLAGS += -j$(JOBS)
  endif
endif

.PHONY: debug release profile simple clean test analyse gprof


GUROBI_HOME ?= $(GUROBI_HOME)
GUROBI_INCLUDE = $(GUROBI_HOME)/include
GUROBI_LIB = $(GUROBI_HOME)/lib

ORTOOLS_HOME ?= /home/matthias/or-tools_x86_64_ubuntu-24.04_cpp_v9.10.4067
ORTOOLS_INCLUDE = $(ORTOOLS_HOME)/include
ORTOOLS_LIB = $(ORTOOLS_HOME)/lib

JSON_INCLUDE = ./nlohmann

INCLUDES = -I$(GUROBI_INCLUDE) -I$(JSON_INCLUDE) -I$(ORTOOLS_INCLUDE)
LIBS = -L$(GUROBI_LIB) -L$(ORTOOLS_LIB)


# Solvers & utilities 
LDLIBS = -lgurobi_c++ -lgurobi130 -llapack -lblas -larmadillo -lortools

# Multithreading
LDLIBS += -lpthread -lm -ldl -lrt -ltbb
 
# a rule for building the program with full debug information
debug: CXXFLAGS += -g3 -Og #  -g0..3 (g default)
debug: scheduler

# a rule for building the program fully optimized and release it
release: CXXFLAGS += -O3 -DNDEBUG -DPARALLEL -fopenmp -fopt-info-vec=vec.opt
release: scheduler

# a rule for profiling the program, pg is for gprof
profile: CXXFLAGS += -pg -O2
profile: scheduler

# simple rule to build the program fast
simple: CXXFLAGS += -Os
simple: scheduler


BASE_DIR = .
SRC_DIR = $(BASE_DIR)/src
BIN_DIR = $(BASE_DIR)/objects

SOURCES = $(SRC_DIR)/utils.cpp $(SRC_DIR)/lookahead_gurobi.cpp $(SRC_DIR)/greedy.cpp $(SRC_DIR)/linprog.cpp $(SRC_DIR)/main.cpp $(SRC_DIR)/lookahead_breakdown.cpp $(SRC_DIR)/traversal.cpp $(SRC_DIR)/traversal_breakdown.cpp
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%.o)


scheduler: $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(BIN_DIR)/scheduler.o $(CXXFLAGS) $(INCLUDES) $(LIBS) $(LDLIBS)

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)



INSTANCES_DIR = ../instances
SOLUTIONS_DIR = ../solutions
PROFILING_DIR = ../profiling

ARGS = --input_file=$(INSTANCES_DIR)/large.json --method=traversal --lookahead 5 --gurobi_threads 12 --output_file=$(SOLUTIONS_DIR)/large.json

clean:
	rm -f $(BIN_DIR)/*.o

test:	
	$(BIN_DIR)/scheduler.o $(ARGS)

analyse:
	$(BIN_DIR)/scheduler.o $(ARGS)
	gprof $(BIN_DIR)/scheduler.o | ./gprof2dot.py | dot -Tpdf -o $(PROFILING_DIR)/output.pdf

gprof:
	$(BIN_DIR)/scheduler.o $(ARGS)
	gprof -q $(BIN_DIR)/scheduler.o gmon.out > $(PROFILING_DIR)/analysis.txt
