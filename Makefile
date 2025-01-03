CXX = g++
CXXFLAGS := -std=c++20 -Wall -Wextra -O3 # -O3

GUROBI_INCLUDE = /opt/gurobi1200/linux64/include
GUROBI_LIB = /opt/gurobi1200/linux64/lib

ORTOOLS_INCLUDE = /opt/or-tools_v9.11.4210/include
ORTOOLS_LIB = /opt/or-tools_v9.11.4210/lib

JSON_INCLUDE = nlohmann

INCLUDES = -I$(GUROBI_INCLUDE) -I$(ORTOOLS_INCLUDE) -I$(JSON_INCLUDE)
LIBS = -L$(GUROBI_LIB) -L$(ORTOOLS_LIB) -lgurobi_c++ -lgurobi120 -lortools

SOURCES = main.cpp utils.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: solve

solve: $(OBJECTS)
	$(CXX) $(OBJECTS) -o solve.o $(CXXFLAGS) $(INCLUDES) $(LIBS)


main.o: main.cpp
	$(CXX) -c main.cpp -o main.o $(CXXFLAGS) $(INCLUDES)

utils.o: utils.cpp
	$(CXX) -c utils.cpp -o utils.o $(CXXFLAGS) $(INCLUDES)

clean:
	rm -f solve *.o

run:	
	./solve


