
include_directories(.)
include_directories(3rd_party)

add_executable(
  amun
  test2/test.cpp
)

SET(EXES "amun")

foreach(exec ${EXES})
  if(CUDA_FOUND)
    target_link_libraries(${exec} ${EXT_LIBS})
    cuda_add_cublas_to_target(${exec})
  else(CUDA_FOUND)
    target_link_libraries(${exec} ${EXT_LIBS})
  endif(CUDA_FOUND)
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach(exec)

add_subdirectory(3rd_party)
