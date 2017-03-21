find_package(Boost 1.59 REQUIRED COMPONENTS unit_test_framework REQUIRED)

include_directories(
	${Boost_INCLUDE_DIRS}
)

add_definitions(-DBOOST_TEST_DYN_LINK)
add_executable(OrthtreeTest orthtree_test.cpp)
target_link_libraries(
	OrthtreeTest
	${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
)

add_test(NAME OrthtreeTest COMMAND OrthtreeTest)

