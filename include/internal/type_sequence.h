#ifndef __NBODY_INTERNAL_TYPE_SEQUENCE_H_
#define __NBODY_INTERNAL_TYPE_SEQUENCE_H_

#include <cstddef>
#include <type_traits>
#include <utility>

namespace nbody {
namespace internal {
namespace typelist {




template<typename List>
struct length;

template<typename T, T... Ts>
struct length<std::integer_sequence<T, Ts...> :
	public std::integral_constant<std::size_t, sizeof...(Ts)> {
};

template<typename List>
constexpr auto length_v = length<List>::value;




template<typename List1, typename List2>
struct equal;

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
			std::integer_sequence<T, Tail2...> > {
};

template<typename List1, typename List2>
constexpr bool equal_v = equal<List1, List2>::value;




template<typename List, std::size_t Index>
struct get;

template<typename T, T Head, T... Tail>
struct get<std::integer_sequence<T, Head, Tail...>, 0> :
	public std::integral_constant<T, Head> {
};

template<typename T, T Head, T... Tail, std::size_t Index>
struct get<std::integer_sequence<T, Head, Tail...>, Index> :
	public get<std::integer_sequence<T, Tail...>, Index - 1> {
};

template<typename List, std::size_t Index>
constexpr auto get_v = get<List, Index>::value;




template<typename List, std::size_t Index, T Value>
struct set;

template<typename T, T Head, T... Tail, T Value>
struct set<std::integer_sequence<T, Head, Tail...>, 0, Value> {
	using type = std::integer_sequence<T, Value, Tail...>;
};

template<typename T, T Head, T... Tail, std::size_t Index, T Value>
struct set<std::integer_sequence<T, Head, Tail...>, Index, Value> :
	public std::integer_sequence<T, Head,
		set_t<std::integer_sequence<T, Tail...>, Index - 1, Value> > {
};

template<typename List, std::size_t Index, T Value>
using set_t = set<List, Index, Value>::type;




template<typename List, std::size_t Index, typename T, T Value>
struct insert;

template<typename T, T... Ts, T Value>
struct insert<std::integer_sequence<T, Ts...>, 0, Value> {
	using type = std::integer_sequence<T, Value, Ts...>;
};

template<typename T, T Head, T... Tail, std::size_t Index, T Value>
struct insert<std::integer_sequence<T, Head, Tail...>, Index, Value> {
	using type = insert_t<
		insert_t<
			std::integer_sequence<T, Tail...>,
			Index - 1,
			T, Value>,
		0,
		T, Head>;
};

template<typename List, std::size_t Index, typename T, T Value>
using insert_t = insert<List, Index, T, Value>::type;




template<typename List, typename T, T Value>
struct count;

template<typename T, T Value>
struct count<std::integer_sequence<T>, T, Value> :
	public std::integral_constant<std::size_t, 0> {
};

template<typename T, T Head, T... Tail, T Value>
struct count<std::integer_sequence<T, Head, Tail...>, T, Value> :
	public std::conditional_t<
		Head == Value,
		std::integral_constant<std::size_t,
			count_v<std::integer_sequence<T, Tail...>, T, Value> >,
		std::integral_constant<std::size_t,
			1 + count_v<std::integer_sequence<T, Tail...>, T, Value> > {
};

template<typename List, typename T, T Value>
constexpr auto count_v = count<List, T, Value>::value;




template<typename List, typename T, T Value>
struct find;

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
			1 + find_v<std::integer_sequence<T, Tail...>, T, Value> > {
};

template<typename List, typename T, T Value>
constexpr auto find_v = find<List, T, Value>::value;




template<typename List, typename T, T Value>
using contains = std::conditional<
	length_v<List> == find_v<List, T, Value>,
	std::true_type,
	std::false_type>;

template<typename List, typename T, T Value>
constexpr bool contains_v = contains<List, T, Value>::value;




template<typename List1, typename List2>
struct merge;

template<typename T, T... Ts1, T... Ts2>
struct merge<
		std::integer_sequence<T, Ts1...>,
		std::integer_sequence<T, Ts2...> > {
	using type = std::integer_sequence<T, Ts1..., Ts2...>;
};

template<typename List1, typename List2>
using merge_t = merge<List1, List2>;




template<typename List, typename T, T Value>
struct insert_into_sorted;

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

template<typename List, typename T, T Value>
using insert_into_sorted_t = insert_into_sorted<List, T, Value>::type;




template<typename List>
struct sort;

template<typename T, T... Ts>
struct sort<std::integer_sequence<T> > {
	using type = std::integer_sequence<T>;
};

template<typename T, T Head, T... Tail>
struct sort<std::integer_sequence<T, Head, Tail...> > {
	using type = insert_into_sorted_t<
		sort_t<std::integer_sequence<T, Tail...> >,
		T, Head>;
};

template<typename List>
using sort_t = sort<List>::type;




template<typename List1, typename List2>
using is_permutation_of = equal<sort_t<List1>, sort_t<List2> >;

template<typename List1, typename List2>
constexpr bool is_permutatation_of_v = is_permutation_of<List1, List2>::value;




}
}
}

#endif

