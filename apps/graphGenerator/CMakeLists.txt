
find_package(Boost)

include_directories(${Boost_INCLUDE_DIRS})
add_executable(graphGenerator main.cpp)
target_compile_features(graphGenerator PRIVATE cxx_std_14)

