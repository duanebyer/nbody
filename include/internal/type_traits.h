#ifndef __NBODY_INTERNAL_TYPE_TRAITS_H_
#define __NBODY_INTERNAL_TYPE_TRAITS_H_

#include <utility>
#include <type_traits>

namespace nbody {
namespace internal {

template<
	typename Result,
	typename F,
	typename... Args,
	typename = std::enable_if_t<
		std::is_convertible<
			decltype(std::declval<F>()(std::declval<Args>()...)),
			Result>::value
	> >
std::true_type is_invocable_r_test(Result, F, Args...);
std::false_type is_invocable_r_test(...);

template<
	typename F,
	typename... Args,
	typename = decltype(std::declval<F>()(std::declval<Args>()...))>
std::true_type is_invocable_test(F, Args...);
std::false_type is_invocable_test(...);

/**
 * \brief This type mimics the C++17 type `std::is_invocable`.
 * 
 * To maintain C++14 compatibility, this custom version is implemented here
 * instead.
 */
template<typename F, typename... Args>
struct is_invocable final : decltype(is_invocable_test(
		std::declval<F>(),
		std::declval<Args>()...)) {
};

template<typename F, typename... Args>
constexpr bool is_invocable_v = is_invocable<F, Args...>::value;

/**
 * \brief This type mimics the C++17 type `std::is_invocable_r`.
 * 
 * To maintain C++14 compatibility, this custom version is implemented here
 * instead.
 */
template<typename Result, typename F, typename... Args>
struct is_invocable_r final : decltype(is_invocable_r_test(
		std::declval<Result>(),
		std::declval<F>(),
		std::declval<Args>()...)) {
};

template<typename Result, typename F, typename... Args>
constexpr bool is_invocable_r_v = is_invocable_r<Result, F, Args...>::value;

}
}

#endif

