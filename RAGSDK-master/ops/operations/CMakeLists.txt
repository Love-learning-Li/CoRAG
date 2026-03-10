file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_library(atb_operations SHARED ${SOURCE_FILES})
target_link_libraries(atb_operations PRIVATE atb opapi)
install(TARGETS atb_operations DESTINATION lib)
