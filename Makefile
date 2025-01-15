CXX = g++
CXXFLAGS := -std=c++20 -Wall -Wextra

OPTIONAL_FLAGS := -Werror -fanalyzer -pedantic
INDIVIDUAL_FLAGS := -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function -Wno-unused-private-field -Wno-unused-value -Wno-unused-local-typedefs 

GUROBI_INCLUDE = /opt/gurobi1200/linux64/include
GUROBI_LIB = /opt/gurobi1200/linux64/lib

ORTOOLS_INCLUDE = /opt/or-tools_v9.11.4210/include
ORTOOLS_LIB = /opt/or-tools_v9.11.4210/lib

JSON_INCLUDE = nlohmann

INCLUDES = -I$(GUROBI_INCLUDE) -I$(ORTOOLS_INCLUDE) -I$(JSON_INCLUDE)
LIBS = -L$(GUROBI_LIB) -L$(ORTOOLS_LIB) -lgurobi_c++ -lgurobi120 -lortools -ltbb -lpthread -lstdc++fs

SOURCES = utils.cpp solve.cpp main.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

INSTANCES_DIR = instances

debug: CXXFLAGS += -g3 -O0  # -g0..3 (g default), and pg is for gprof
debug: scheduler

build: CXXFLAGS += -O3 -DNDEBUG
build: scheduler

profile: CXXFLAGS += -pg -O3
profile: scheduler

# simple rule to build the program
simple: CXXFLAGS += -Os
simple: scheduler


scheduler: $(OBJECTS)
	$(CXX) $(OBJECTS) -o scheduler.o $(CXXFLAGS) $(INCLUDES) $(LIBS)


main.o: main.cpp
	$(CXX) -c main.cpp -o main.o $(CXXFLAGS) $(INCLUDES)

utils.o: utils.cpp
	$(CXX) -c utils.cpp -o utils.o $(CXXFLAGS) $(INCLUDES)

solve.o: solve.cpp
	$(CXX) -c solve.cpp -o solve.o $(CXXFLAGS) $(INCLUDES)

clean:
	rm -f *.o

test:	
	./scheduler.o $(INSTANCES_DIR)/tiny.json --lookahead=10 --gurobi_threads=2 --time_limit=10.0

analyse:
	./scheduler.o $(INSTANCES_DIR)/tiny.json --lookahead=10 --gurobi_threads=1 --time_limit=10.0 && gprof -q ./solve.o gmon.out > analysis.txt