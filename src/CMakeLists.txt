



file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cu" "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Make an automatic library - will be static or dynamic based on user setting
add_library(shortest_path_library ${CUDA_SOURCES} )

set_target_properties(shortest_path_library PROPERTIES
  INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/common/nvidia/"
  CUDA_SEPARABLE_COMPILATION OFF
)
# All users of this library will need at least C++11
target_compile_features(shortest_path_library PUBLIC cxx_std_14)
target_include_directories(shortest_path_library PUBLIC ${HEADER_LIST})

target_link_libraries(shortest_path_library ${CUDA_LIBRARIES})

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}/common" PREFIX "Nvidia Common Files" FILES ${HEADER_LIST})
