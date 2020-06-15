
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories("${PROJECT_SOURCE_DIR}/common/our/")
include_directories("${PROJECT_SOURCE_DIR}/common/nvidia/")

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h"  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB COMMON_FILES "${PROJECT_SOURCE_DIR}/common/*/*.h" )


add_executable(floydWarshallCuda ${SOURCES} ${COMMON_FILES})


set_target_properties(floydWarshallCuda PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(floydWarshallCuda PRIVATE cxx_std_14)
target_link_libraries(floydWarshallCuda  Boost::filesystem )

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}" PREFIX "Common Files" FILES ${COMMON_FILES})

install(TARGETS floydWarshallCuda
        CONFIGURATIONS Release
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/release)
