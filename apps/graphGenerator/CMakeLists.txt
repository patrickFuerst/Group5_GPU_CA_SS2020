
find_package(Boost)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories("${PROJECT_SOURCE_DIR}/common/our/")

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB COMMON_FILES "${PROJECT_SOURCE_DIR}/common/our/*.h" )

add_executable(graphGenerator ${SOURCES} ${COMMON_FILES} )

target_compile_features(graphGenerator PRIVATE cxx_std_14)
target_link_libraries(graphGenerator  Boost::filesystem)

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}" PREFIX "Common Files" FILES ${COMMON_FILES})

install(TARGETS graphGenerator
        CONFIGURATIONS Release
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/release)
