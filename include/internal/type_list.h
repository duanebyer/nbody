#ifndef __NBODY_INTERNAL_TYPE_LIST_H_
#define __NBODY_INTERNAL_TYPE_LIST_H_

#include <cstddef>
#include <type_traits>
#include <utility>

namespace nbody {
namespace internal {
namespace typelist {

/**
 * \brief Type-level function that calculates the length of a list.
 */
template<typename List>
struct length;
template<typename List>
constexpr auto length_v = length<List>::value;

template<typename T, T... Ts>
struct length<std::integer_sequence<T, Ts...> > :
	public std::integral_constant<std::size_t, sizeof...(Ts)> {
};



/**
 * \brief Type-level function that determines if two lists are equal.
 */
template<typename List1, typename List2>
struct equal;
template<typename List1, typename List2>
constexpr bool equal_v = equal<List1, List2>::value;

template<typename T>
struct equal<
		std::integer_sequence<T>,
		std::integer_sequence<T> > :
	public std::true_type {
};
template<typename T, T... Ts>
struct equal<
		std::integer_sequence<T, Ts...>,
		std::integer_sequence<T> > :
	public std::false_type {
};
template<typename T, T... Ts>
struct equal<
		std::integer_sequence<T>,
		std::integer_sequence<T, Ts...> > :
	public std::false_type {
};
template<typename T, T Head1, T... Tail1, T Head2, T... Tail2>
struct equal<
		std::integer_sequence<T, Head1, Tail1...>,
		std::integer_sequence<T, Head2, Tail2...> > :
	public std::conditional_t<
		Head1 != Head2,
		std::false_type,
		equal<
			std::integer_sequence<T, Tail1...>,
			std::integer_sequence<T, Tail2...> > > {
};



/**
 * \brief Type-level function that inserts an element into a list.
 */
template<
	typename List, std::size_t Index, typename T, T Value,
	typename Enable = void>
struct insert;
template<typename List, std::size_t Index, typename T, T Value>
using insert_t = typename insert<List, Index, T, Value>::type;

template<typename T, T... Ts, T Value>
struct insert<std::integer_sequence<T, Ts...>, 0, T, Value> {
	using type = std::integer_sequence<T, Value, Ts...>;
};
template<typename T, T Head, T... Tail, std::size_t Index, T Value>
struct insert<
		std::integer_sequence<T, Head, Tail...>, Index, T, Value,
		std::enable_if_t<Index != 0> > {
	using type = insert_t<
		insert_t<
			std::integer_sequence<T, Tail...>,
			Index - 1,
			T, Value>,
		0,
		T, Head>;
};



/**
 * \brief Type-level function that gets the element from a certain index in a
 * list.
 */
template<
	typename List, std::size_t Index,
	typename Enable = void>
struct get;
template<typename List, std::size_t Index>
constexpr auto get_v = get<List, Index>::value;

template<typename T, T Head, T... Tail>
struct get<std::integer_sequence<T, Head, Tail...>, 0> :
	public std::integral_constant<T, Head> {
};
template<typename T, T Head, T... Tail, std::size_t Index>
struct get<
		std::integer_sequence<T, Head, Tail...>, Index,
		std::enable_if_t<Index != 0> > :
	public get<std::integer_sequence<T, Tail...>, Index - 1> {
};



/**
 * \brief Type-level function that sets an element at a certain index of a list.
 */
template<
	typename List, std::size_t Index, typename T, T Value,
	typename Enable = void>
struct set;
template<typename List, std::size_t Index, typename T, T Value>
using set_t = typename set<List, Index, T, Value>::type;

template<typename T, T Head, T... Tail, T Value>
struct set<std::integer_sequence<T, Head, Tail...>, 0, T, Value> {
	using type = std::integer_sequence<T, Value, Tail...>;
};
template<typename T, T Head, T... Tail, std::size_t Index, T Value>
struct set<
		std::integer_sequence<T, Head, Tail...>, Index, T, Value,
		std::enable_if_t<Index != 0> > {
	using type = insert_t<
		set_t<std::integer_sequence<T, Tail...>, Index - 1, T, Value>,
		0,
		T, Head>;
};



/**
 * \brief Type-level function that determines how many times a certain element
 * appears in a list.
 */
template<typename List, typename T, T Value>
struct count;
template<typename List, typename T, T Value>
constexpr auto count_v = count<List, T, Value>::value;

template<typename T, T Value>
struct count<std::integer_sequence<T>, T, Value> :
	public std::integral_constant<std::size_t, 0> {
};
template<typename T, T Head, T... Tail, T Value>
struct count<std::integer_sequence<T, Head, Tail...>, T, Value> :
	public std::conditional_t<
		Head == Value,
		std::integral_constant<std::size_t,
			1 + count_v<std::integer_sequence<T, Tail...>, T, Value> >,
		std::integral_constant<std::size_t,
			0 + count_v<std::integer_sequence<T, Tail...>, T, Value> > > {
};



/**
 * \brief Type-level function that finds the index of the first occurence of an
 * element in a list.
 */
template<typename List, typename T, T Value>
struct find;
template<typename List, typename T, T Value>
constexpr auto find_v = find<List, T, Value>::value;

template<typename T, T Value>
struct find<std::integer_sequence<T>, T, Value> :
	public std::integral_constant<std::size_t, 0> {
};
template<typename T, T Head, T... Tail, T Value>
struct find<std::integer_sequence<T, Head, Tail...>, T, Value> :
	public std::conditional_t<
		Head == Value,
		std::integral_constant<std::size_t, 0>,
		std::integral_constant<std::size_t,
			1 + find_v<std::integer_sequence<T, Tail...>, T, Value> > > {
};



/**
 * \brief Type-level function that determines if a list contains a certain
 * element.
 */
template<typename List, typename T, T Value>
using contains = std::conditional_t<
	length_v<List> != find_v<List, T, Value>,
	std::true_type,
	std::false_type>;
template<typename List, typename T, T Value>
constexpr bool contains_v = contains<List, T, Value>::value;



/**
 * \brief Type-level function that determines if every element in a list is
 * unique.
 */
template<typename List>
struct unique;
template<typename List>
constexpr bool unique_v = unique<List>::value;

template<typename T>
struct unique<std::integer_sequence<T> > : public std::true_type {
};
template<typename T, T Head, T... Tail>
struct unique<std::integer_sequence<T, Head, Tail...> > :
	public std::conditional_t<
		contains_v<std::integer_sequence<T, Tail...>, T, Head>,
		std::false_type,
		unique<std::integer_sequence<T, Tail...> > > {
};



/**
 * \brief Type-level function that concatenates two lists together.
 */
template<typename List1, typename List2>
struct merge;
template<typename List1, typename List2>
using merge_t = typename merge<List1, List2>::type;

template<typename T, T... Ts1, T... Ts2>
struct merge<
		std::integer_sequence<T, Ts1...>,
		std::integer_sequence<T, Ts2...> > {
	using type = std::integer_sequence<T, Ts1..., Ts2...>;
};



/**
 * \brief Type-level function that inserts an element into the appropriate
 * location in an already-sorted list.
 */
template<typename List, typename T, T Value>
struct insert_into_sorted;
template<typename List, typename T, T Value>
using insert_into_sorted_t = typename insert_into_sorted<List, T, Value>::type;

template<typename T, T Value>
struct insert_into_sorted<std::integer_sequence<T>, T, Value> {
	using type = std::integer_sequence<T, Value>;
};
template<typename T, T Head, T... Tail, T Value>
struct insert_into_sorted<std::integer_sequence<T, Head, Tail...>, T, Value> {
	using type = std::conditional_t<
		Value <= Head,
		std::integer_sequence<T, Value, Head, Tail...>,
		insert_t<
			insert_into_sorted_t<std::integer_sequence<T, Tail...>, T, Value>,
			0,
			T, Head> >;
};



/**
 * \brief Type-level function that sorts a list.
 */
template<typename List>
struct sort;
template<typename List>
using sort_t = typename sort<List>::type;

template<typename T>
struct sort<std::integer_sequence<T> > {
	using type = std::integer_sequence<T>;
};
template<typename T, T Head, T... Tail>
struct sort<std::integer_sequence<T, Head, Tail...> > {
	using type = insert_into_sorted_t<
		sort_t<std::integer_sequence<T, Tail...> >,
		T, Head>;
};



/**
 * \brief Type-level function that determines if one list is a permutation of
 * another list.
 */
template<typename List1, typename List2>
using is_permutation_of = equal<sort_t<List1>, sort_t<List2> >;
template<typename List1, typename List2>
constexpr bool is_permutation_of_v = is_permutation_of<List1, List2>::value;

}
}
}

#endif

