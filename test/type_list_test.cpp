#define BOOST_TEST_MODULE TypeListTest

#include <boost/test/unit_test.hpp>
#include <boost/utility/identity_type.hpp>

#include <utility>

#include "internal/type_list.h"

// These macros are used so that template functions can be called inside the
// Boost test macros without causing compile errors (because of the commas). For
// example, BOOST_CHECK(equal_v<list, list>) would be a compile error due to the
// commas in the template parameters.
#define IDENTITY_TYPE(x) BOOST_IDENTITY_TYPE(x)
#define IDENTITY_VAL(x) (x)

using namespace nbody::internal::typelist;

// A set of type level lists that are used to test various cases.
using emptyList = std::integer_sequence<int>;
using singletonList = std::integer_sequence<int, 4>;
using list = std::integer_sequence<
	int, 6, 8, 3, 4, 6, 5, 5, 4, 2, 4, 8, 9>;
using insertStartList = std::integer_sequence<
	int, 0, 6, 8, 3, 4, 6, 5, 5, 4, 2, 4, 8, 9>;
using insertMiddleList = std::integer_sequence<
	int, 6, 8, 3, 4, 6, 5, 0, 5, 4, 2, 4, 8, 9>;
using insertEndList = std::integer_sequence<
	int, 6, 8, 3, 4, 6, 5, 5, 4, 2, 4, 8, 9, 0>;
using setStartList = std::integer_sequence<
	int, 0, 8, 3, 4, 6, 5, 5, 4, 2, 4, 8, 9>;
using setMiddleList = std::integer_sequence<
	int, 6, 8, 3, 4, 6, 5, 0, 4, 2, 4, 8, 9>;
using setEndList = std::integer_sequence<
	int, 6, 8, 3, 4, 6, 5, 5, 4, 2, 4, 8, 0>;
using mergeStartList = std::integer_sequence<
	int, 4, 6, 8, 3, 4, 6, 5, 5, 4, 2, 4, 8, 9>;
using mergeEndList = std::integer_sequence<
	int, 6, 8, 3, 4, 6, 5, 5, 4, 2, 4, 8, 9, 4>;
using mergedList = std::integer_sequence<
	int, 6, 8, 3, 4, 6, 5, 5, 4, 2, 4, 8, 9, 6, 3, 4, 8, 9, 1, 2, 7, 5, 0>;
using sortedList = std::integer_sequence<
	int, 2, 3, 4, 4, 4, 5, 5, 6, 6, 8, 8, 9>;
using reverseList = std::integer_sequence<
	int, 9, 8, 8, 6, 6, 5, 5, 4, 4, 4, 3, 2>;
using permutedList = std::integer_sequence<
	int, 8, 5, 4, 2, 8, 9, 5, 4, 6, 4, 6, 3>;
using uniqueList = std::integer_sequence<
	int, 6, 3, 4, 8, 9, 1, 2, 7, 5, 0>;

BOOST_AUTO_TEST_CASE(TypeListLengthTest) {
	BOOST_CHECK_EQUAL(length_v<emptyList>, 0);
	BOOST_CHECK_EQUAL(length_v<singletonList>, 1);
	BOOST_CHECK_EQUAL(length_v<list>, 12);
	BOOST_CHECK_EQUAL(length_v<uniqueList>, 10);
}

BOOST_AUTO_TEST_CASE(TypeListEqualTest) {
	BOOST_CHECK(IDENTITY_VAL((equal_v<emptyList, emptyList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<singletonList, singletonList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<list, list>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<uniqueList, uniqueList>)));
	
	BOOST_CHECK(IDENTITY_VAL((!equal_v<list, emptyList>)));
	BOOST_CHECK(IDENTITY_VAL((!equal_v<list, singletonList>)));
	BOOST_CHECK(IDENTITY_VAL((!equal_v<list, permutedList>)));
	BOOST_CHECK(IDENTITY_VAL((!equal_v<list, uniqueList>)));
	
	BOOST_CHECK(IDENTITY_VAL((!equal_v<emptyList, singletonList>)));
}

BOOST_AUTO_TEST_CASE(TypeListInsertTest) {
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		insert_t<emptyList, 0, int, 4>,
		singletonList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		insert_t<list, 0, int, 0>,
		insertStartList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		insert_t<list, 6, int, 0>,
		insertMiddleList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		insert_t<list, 12, int, 0>,
		insertEndList>)));
}

BOOST_AUTO_TEST_CASE(TypeListGetTest) {
	BOOST_CHECK_EQUAL(IDENTITY_VAL((get_v<singletonList, 0>)), 4);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((get_v<list, 0>)), 6);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((get_v<list, 6>)), 5);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((get_v<list, 11>)), 9);
}

BOOST_AUTO_TEST_CASE(TypeListSetTest) {
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		set_t<singletonList, 0, int, 5>,
		std::integer_sequence<int, 5> >)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		set_t<list, 0, int, 0>,
		setStartList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		set_t<list, 6, int, 0>,
		setMiddleList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		set_t<list, 11, int, 0>,
		setEndList>)));
}

BOOST_AUTO_TEST_CASE(TypeListCountTest) {
	BOOST_CHECK_EQUAL(IDENTITY_VAL((count_v<emptyList, int, 1>)), 0);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((count_v<singletonList, int, 1>)), 0);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((count_v<singletonList, int, 4>)), 1);
	
	BOOST_CHECK_EQUAL(IDENTITY_VAL((count_v<list, int, 1>)), 0);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((count_v<list, int, 2>)), 1);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((count_v<list, int, 4>)), 3);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((count_v<list, int, 5>)), 2);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((count_v<list, int, 6>)), 2);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((count_v<list, int, 9>)), 1);
}

BOOST_AUTO_TEST_CASE(TypeListFindTest) {
	BOOST_CHECK_EQUAL(IDENTITY_VAL((find_v<emptyList, int, 1>)), 0);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((find_v<singletonList, int, 1>)), 1);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((find_v<singletonList, int, 4>)), 0);
	
	BOOST_CHECK_EQUAL(IDENTITY_VAL((find_v<list, int, 1>)), 12);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((find_v<list, int, 2>)), 8);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((find_v<list, int, 4>)), 3);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((find_v<list, int, 5>)), 5);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((find_v<list, int, 6>)), 0);
	BOOST_CHECK_EQUAL(IDENTITY_VAL((find_v<list, int, 9>)), 11);
}

BOOST_AUTO_TEST_CASE(TypeListContainsTest) {
	BOOST_CHECK(IDENTITY_VAL((!contains_v<emptyList, int, 1>)));
	BOOST_CHECK(IDENTITY_VAL((!contains_v<singletonList, int, 1>)));
	BOOST_CHECK(IDENTITY_VAL((contains_v<singletonList, int, 4>)));
	
	BOOST_CHECK(IDENTITY_VAL((!contains_v<list, int, 1>)));
	BOOST_CHECK(IDENTITY_VAL((contains_v<list, int, 6>)));
	BOOST_CHECK(IDENTITY_VAL((contains_v<list, int, 5>)));
	BOOST_CHECK(IDENTITY_VAL((contains_v<list, int, 9>)));
}

BOOST_AUTO_TEST_CASE(TypeListMergeTest) {
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		merge_t<emptyList, emptyList>,
		emptyList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		merge_t<emptyList, singletonList>,
		singletonList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		merge_t<singletonList, emptyList>,
		singletonList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		merge_t<emptyList, list>,
		list>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		merge_t<list, emptyList>,
		list>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		merge_t<singletonList, list>,
		mergeStartList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		merge_t<list, singletonList>,
		mergeEndList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		merge_t<list, uniqueList>,
		mergedList>)));
}

BOOST_AUTO_TEST_CASE(TypeListSortTest) {
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		sort_t<emptyList>,
		emptyList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		sort_t<singletonList>,
		singletonList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		sort_t<sortedList>,
		sortedList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		sort_t<reverseList>,
		sortedList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		sort_t<permutedList>,
		sortedList>)));
	BOOST_CHECK(IDENTITY_VAL((equal_v<
		sort_t<list>,
		sortedList>)));
}

BOOST_AUTO_TEST_CASE(TypeListIsPermutationOfTest) {
	BOOST_CHECK(IDENTITY_VAL((is_permutation_of_v<
		emptyList,
		emptyList>)));
	BOOST_CHECK(IDENTITY_VAL((!is_permutation_of_v<
		emptyList,
		singletonList>)));
	BOOST_CHECK(IDENTITY_VAL((is_permutation_of_v<
		singletonList,
		singletonList>)));
	BOOST_CHECK(IDENTITY_VAL((!is_permutation_of_v<
		singletonList,
		std::integer_sequence<int, 5> >)));
	BOOST_CHECK(IDENTITY_VAL((!is_permutation_of_v<
		emptyList,
		list>)));
	BOOST_CHECK(IDENTITY_VAL((!is_permutation_of_v<
		singletonList,
		list>)));
	
	BOOST_CHECK(IDENTITY_VAL((is_permutation_of_v<
		list,
		permutedList>)));
	BOOST_CHECK(IDENTITY_VAL((is_permutation_of_v<
		list,
		sortedList>)));
	BOOST_CHECK(IDENTITY_VAL((is_permutation_of_v<
		list,
		reverseList>)));
	BOOST_CHECK(IDENTITY_VAL((!is_permutation_of_v<
		list,
		uniqueList>)));
}

