CXX = g++
CXXFLAGS := -std=c++20 -Wall -Wextra -march=native

OPTIONAL_FLAGS := -Werror -fanalyzer -pedantic
INDIVIDUAL_FLAGS := -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function -Wno-unused-private-field -Wno-unused-value -Wno-unused-local-typedefs 

GUROBI_INCLUDE = /opt/gurobi1200/linux64/include
GUROBI_LIB = /opt/gurobi1200/linux64/lib

ORTOOLS_INCLUDE = /opt/or-tools_v9.11.4210/include
ORTOOLS_LIB = /opt/or-tools_v9.11.4210/lib

JSON_INCLUDE = ./nlohmann

INCLUDES = -I$(GUROBI_INCLUDE) -I$(ORTOOLS_INCLUDE) -I$(JSON_INCLUDE)
LIBS = -L$(GUROBI_LIB) -L$(ORTOOLS_LIB) -lgurobi_c++ -lgurobi120 -lortools -ltbb -lpthread -lstdc++fs




# a rule for building the program with full debug information; -g0..3 (g default), and pg is for gprof
debug: CXXFLAGS += -g3 -O0
debug: scheduler

# a rule for building the program fully optimized and release it
build: CXXFLAGS += -O3 -DNDEBUG -fopt-info-vec -fopenmp
build: scheduler

# a rule for profiling the program
profile: CXXFLAGS += -pg -O3
profile: scheduler

# simple rule to build the program fast
simple: CXXFLAGS += -Os
simple: scheduler


SRC_DIR = cpp_solver
BIN_DIR = $(SRC_DIR)/objects

SOURCES = $(SRC_DIR)/utils.cpp $(SRC_DIR)/solve.cpp $(SRC_DIR)/main.cpp $(SRC_DIR)/breakdown.cpp 
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%.o)

INSTANCES_DIR = instances
PROFILING_DIR = profiling

scheduler: $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(BIN_DIR)/scheduler.o $(CXXFLAGS) $(INCLUDES) $(LIBS)

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

clean:
	rm -f $(BIN_DIR)/*.o

test:	
	$(BIN_DIR)/scheduler.o $(INSTANCES_DIR)/tiny.json --lookahead=10 --gurobi_threads=4 --time_limit=10.0

analyse:
	$(BIN_DIR)/scheduler.o $(INSTANCES_DIR)/tiny.json --lookahead=10 --gurobi_threads=1 --time_limit=10.0
	gprof $(BIN_DIR)/scheduler.o | ./gprof2dot.py | dot -Tpdf -o $(PROFILING_DIR)/output.pdf

gprof:
	$(BIN_DIR)/scheduler.o $(INSTANCES_DIR)/tiny.json --lookahead=10 --gurobi_threads=1 --time_limit=10.0 && gprof -q $(BIN_DIR)/scheduler.o gmon.out > $(PROFILING_DIR)/analysis.txt
