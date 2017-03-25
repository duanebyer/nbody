#ifndef __NBODY_INTERNAL_FUNCTIONAL_H_
#define __NBODY_INTERNAL_FUNCTIONAL_H_

#include <climits>

namespace nbody {
namespace internal {

/**
 * \brief A function type that wraps the array subscript operator.
 */
template<typename T>
struct subscript final {
	auto& operator()(T& val, std::size_t index) const {
		return val[index];
	}
	auto const& operator()(T const& val, std::size_t index) const {
		return val[index];
	}
};

template<>
struct subscript<void> final {
	template<typename T>
	auto& operator()(T& val, std::size_t index) const {
		return val[index];
	}
	template<typename T>
	auto operator()(T const& val, std::size_t index) const {
		return val[index];
	}
};

}
}

#endif

