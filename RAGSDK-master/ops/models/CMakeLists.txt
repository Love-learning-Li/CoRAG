file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_library(atb_models SHARED ${SOURCE_FILES})
target_link_libraries(atb_models PRIVATE atb atb_operations)
install(TARGETS atb_models DESTINATION lib)
