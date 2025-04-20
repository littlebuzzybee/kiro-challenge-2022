cmake_minimum_required(VERSION 3.20)
project(SchedulerSolver CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler warnings and flags
add_compile_options(-Wall -Wextra -march=native)

# Optional and individual flags (you can toggle these as needed)
add_compile_options(
    -Wno-unused-parameter 
    -Wno-unused-variable 
    -Wno-unused-but-set-variable 
    -Wno-unused-function 
    -Wno-unused-private-field 
    -Wno-unused-value 
    -Wno-unused-local-typedefs
)

# External libraries
set(GUROBI_INCLUDE /opt/gurobi1200/linux64/include)
set(GUROBI_LIB /opt/gurobi1200/linux64/lib)

set(ORTOOLS_INCLUDE /opt/or-tools_v9.11.4210/include)
set(ORTOOLS_LIB /opt/or-tools_v9.11.4210/lib)

set(HIGHS_INCLUDE /opt/highs-1.10.0/include/highs)
set(HIGHS_LIB /opt/highs-1.10.0/lib)

set(JSON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann)

include_directories(
    ${GUROBI_INCLUDE}
    ${ORTOOLS_INCLUDE}
    ${HIGHS_INCLUDE}
    ${JSON_INCLUDE}
)

link_directories(
    ${GUROBI_LIB}
    ${ORTOOLS_LIB}
    ${HIGHS_LIB}
)

# Source files
file(GLOB_RECURSE SRC_FILES
    utils.cpp
    solve_gurobi.cpp
    solve_greedy.cpp
    solve_linprog.cpp
    main.cpp
    breakdown.cpp
    solve_traversal.cpp
)

add_executable(scheduler ${SRC_FILES})

# Linked libraries
target_link_libraries(scheduler
    gurobi_c++
    gurobi120
    tbb
    pthread
    stdc++fs
    ortools
    lapack
    blas
    armadillo
    highs
)

# Build types
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Os -DNDEBUG -mavx -mavx2 -msse -msse3 -msse4.1 -mssse3 -msse2 -fopenmp -fopt-info-vec")
set(CMAKE_CXX_FLAGS_PROFILE "-pg -O2")
set(CMAKE_CXX_FLAGS_SIMPLE "-Os -O1")

# Output directory
set_target_properties(scheduler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/objects
)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
