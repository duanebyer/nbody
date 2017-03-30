#ifndef __NBODY_TENSOR_TENSOR_INDEX_EXPRESSION_H_
#define __NBODY_TENSOR_TENSOR_INDEX_EXPRESSION_H_

#include <type_traits>
#include <utility>

#include "tensor.h"

namespace nbody {

enum class TensorIndex {
	All,
	A, B, C, D, E, F, G,
	H, I, J, K, L, M, N, O, P,
	Q, R, S, T, U, V,
	W, X, Y, Z,
	// Now I know my A, B, Cs,...
};

template<
	typename UpperIndices,
	typename LowerIndices,
	std::size_t Dim,
	typename Scalar>
class TensorIndexExpression;

template<
	TensorIndex... UpperIndices,
	TensorIndex... LowerIndices,
	std::size_t Dim,
	typename Scalar>
class TensorIndexExpression<
		std::integer_sequence<TensorIndex, UpperIndices...>,
		std::integer_sequence<TensorIndex, LowerIndices...>,
		Dim,
		Scalar> final {
	
private:
	
	using TensorType = Tensor<
		sizeof...(UpperIndices),
		sizeof...(LowerIndices),
		Dim,
		Scalar>;
	
	TensorType _tensor;
	
	TensorIndexExpression(
			TensorType tensor,
			_tensor(tensor) {
	}
	
public:
	
};

}

#endif

