find_package(Boost 1.59 REQUIRED COMPONENTS unit_test_framework REQUIRED)

include_directories(
	${Boost_INCLUDE_DIRS}
)

add_definitions(-DBOOST_TEST_DYN_LINK)
add_executable(OrthtreeTest orthtree_test.cpp)
add_executable(TypeListTest type_list_test.cpp)
target_link_libraries(
	TypeListTest
	${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
)
target_link_libraries(
	OrthtreeTest
	${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
)

add_test(NAME OrthtreeTest COMMAND OrthtreeTest)
add_test(Name TypeListTest COMMAND TypeListTest)

