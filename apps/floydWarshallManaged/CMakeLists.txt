
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories("${PROJECT_SOURCE_DIR}/common/our/")
include_directories("${PROJECT_SOURCE_DIR}/common/nvidia/")

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h"  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB COMMON_FILES "${PROJECT_SOURCE_DIR}/common/our/*.h" )


add_executable(floydWarshallManaged ${SOURCES} ${COMMON_FILES})


set_target_properties(floydWarshallManaged PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
   CUDA_STANDARD 11
)

target_compile_features(floydWarshallManaged PRIVATE cxx_std_14)
target_link_libraries(floydWarshallManaged Boost::filesystem )

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}" PREFIX "Common Files" FILES ${COMMON_FILES})

install(TARGETS floydWarshallManaged
        CONFIGURATIONS Release
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/release)
