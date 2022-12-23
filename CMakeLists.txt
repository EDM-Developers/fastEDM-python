cmake_minimum_required(VERSION 3.4...3.18)
project(_fastEDM)


# Write compile_commands.json file (used by IDEs & static analysers)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Compiler flags
if(WIN32)
  set(CMAKE_CXX_FLAGS_DEBUG "/MTd /W4 /permissive /ZI /fsanitize=address /bigobj")
  set(CMAKE_CXX_FLAGS_RELEASE "/MT /Ox /fp:fast /external:anglebrackets /external:W0")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
endif()

set(COMMON_SRC src/common.cpp src/common.h
               src/cpu.cpp src/cpu.h
               src/distances.cpp src/distances.h
               src/edm.cpp src/edm.h
               src/manifold.cpp src/manifold.h
               src/stats.cpp src/stats.h
               src/library_prediction_split.h
               src/vendor/mersennetwister.h
               src/vendor/core.h src/vendor/EMD_wrapper.h
               src/vendor/network_simplex_simple.h src/vendor/full_bipartitegraph.h
               src/vendor/thread_pool.h
               )

add_subdirectory(extern/pybind11)
pybind11_add_module(_fastEDM ${COMMON_SRC} src/python.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_fastEDM
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

target_include_directories(_fastEDM PRIVATE src/vendor)

target_compile_definitions(_fastEDM PRIVATE JSON)
target_compile_definitions(_fastEDM PRIVATE NOMINMAX)

set_target_properties(_fastEDM
PROPERTIES
  CXX_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
)