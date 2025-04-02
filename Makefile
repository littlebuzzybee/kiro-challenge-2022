CXX = g++
CXXFLAGS := -std=c++20 -Wall -Wextra -march=native

OPTIONAL_FLAGS := -Werror -fanalyzer -pedantic
INDIVIDUAL_FLAGS := -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function -Wno-unused-private-field -Wno-unused-value -Wno-unused-local-typedefs 

GUROBI_INCLUDE = /opt/gurobi1200/linux64/include
GUROBI_LIB = /opt/gurobi1200/linux64/lib

ORTOOLS_INCLUDE = /opt/or-tools_v9.11.4210/include
ORTOOLS_LIB = /opt/or-tools_v9.11.4210/lib

# KALIS_INCLUDE
# KALIS_LIB

ARMADILLO_LIB = 

JSON_INCLUDE = ./nlohmann

INCLUDES = -I$(GUROBI_INCLUDE) -I$(JSON_INCLUDE) -I$(ORTOOLS_INCLUDE)
LIBS = -L$(GUROBI_LIB) -L$(GUROBI_INCLUDE) -L$(ORTOOLS_LIB)


# Library linking
LDLIBS = -lgurobi_c++ -lgurobi120 -ltbb -lpthread -lstdc++fs -lortools # -larmadillo




# a rule for building the program with full debug information
debug: CXXFLAGS += -g3 -O0 #  -g0..3 (g default), pg is for gprof
debug: scheduler

# a rule for building the program fully optimized and release it
build: CXXFLAGS += -O3 -DNDEBUG -fopt-info-vec -mavx -mavx2 -msse -msse3 -msse4.1 -mssse3 -msse2 # -fopenmp
build: scheduler

# a rule for profiling the program
profile: CXXFLAGS += -pg -O2
profile: scheduler

# simple rule to build the program fast
simple: CXXFLAGS += -Os
simple: scheduler


SRC_DIR = cpp_solver
BIN_DIR = $(SRC_DIR)/objects

SOURCES = $(SRC_DIR)/utils.cpp $(SRC_DIR)/solve_gurobi.cpp $(SRC_DIR)/solve_heuristic.cpp $(SRC_DIR)/solve_ortools.cpp $(SRC_DIR)/main.cpp $(SRC_DIR)/breakdown.cpp 
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%.o)


scheduler: $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(BIN_DIR)/scheduler.o $(CXXFLAGS) $(INCLUDES) $(LIBS) $(LDLIBS)

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)



INSTANCES_DIR = instances
PROFILING_DIR = profiling

clean:
	rm -f $(BIN_DIR)/*.o

test:	
	$(BIN_DIR)/scheduler.o $(INSTANCES_DIR)/tiny.json --lookahead=10 --gurobi_threads=4 --time_limit=10.0

analyse:
	$(BIN_DIR)/scheduler.o $(INSTANCES_DIR)/tiny.json --lookahead=10 --gurobi_threads=1 --time_limit=10.0
	gprof $(BIN_DIR)/scheduler.o | ./gprof2dot.py | dot -Tpdf -o $(PROFILING_DIR)/output.pdf

gprof:
	$(BIN_DIR)/scheduler.o $(INSTANCES_DIR)/tiny.json --lookahead=10 --gurobi_threads=1 --time_limit=10.0 && gprof -q $(BIN_DIR)/scheduler.o gmon.out > $(PROFILING_DIR)/analysis.txt
