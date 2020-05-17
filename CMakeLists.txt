cmake_minimum_required(VERSION 3.17)

# set the project name
project(Group5_GPU_CA_SS2020 
		VERSION 0.1 
		LANGUAGES CUDA CXX)


# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# The compiled library code is here
add_subdirectory("src")

# The executable code is here
add_subdirectory("apps/shortestPathFinder")
add_subdirectory("apps/graphGenerator")
add_subdirectory("apps/shortestPathFinderThrust")
