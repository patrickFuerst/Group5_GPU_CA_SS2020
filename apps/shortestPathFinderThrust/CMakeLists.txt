

add_executable(shortestPathThrust main.cu)


set_target_properties(shortestPath PROPERTIES
  INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/common/nvidia/"
   CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(shortestPathThrust PRIVATE shortest_path_library )

target_compile_features(shortestPathThrust PRIVATE cxx_std_14)

