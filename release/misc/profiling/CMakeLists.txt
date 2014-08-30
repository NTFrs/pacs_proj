#
#  CMake script for the profiling program:
##

include_directories(../../include)

# Set the name of the project and target:
SET(TARGET "main_profiling")

# Declare all source files the target consists of:
SET(TARGET_SRC
  ${TARGET}.cpp
  # You can specify additional files here!
  ../../src/QuadratureRule.cpp
  )
  
# Usually, you will not need to modify anything beyond this point...

ADD_DEFINITIONS(
    -std=c++11 # Or -std=c++0x
    # Other flags
#     -pg
)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})

SET(CLEAN_UP_FILES
  # a custom list of globs, e.g. *.log *.vtk
  ${CLEAN_UP_FILES} matlab/*m plot/*eps gnuplot/*gpl
)

# find_package(OpenMP)
# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()

# set (CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")

DEAL_II_INVOKE_AUTOPILOT()
