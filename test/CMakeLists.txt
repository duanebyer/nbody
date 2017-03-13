find_package(Boost 1.59 REQUIRED COMPONENTS unit_test_framework REQUIRED)

include_directories(
	${Boost_INCLUDE_DIRS}
)

add_definitions(-DBOOST_TEST_DYN_LINK)
add_executable(OctreeTest octree_test.cpp)
target_link_libraries(
	OctreeTest
	${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
)

add_test(NAME OctreeTest COMMAND OctreeTest)

