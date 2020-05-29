


include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories("${PROJECT_SOURCE_DIR}/common/our/")

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h"  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB COMMON_FILES "${PROJECT_SOURCE_DIR}/common/our/*.h" )


add_executable(shortestPathThrust ${SOURCES} ${COMMON_FILES})


set_target_properties(shortestPath PROPERTIES
  INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/common/nvidia/"
   CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(shortestPathThrust PRIVATE shortest_path_library Boost::filesystem )

target_compile_features(shortestPathThrust PRIVATE cxx_std_14)

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}" PREFIX "Common Files" FILES ${COMMON_FILES})
