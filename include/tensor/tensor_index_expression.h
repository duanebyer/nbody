#ifndef __NBODY_TENSOR_TENSOR_INDEX_EXPRESSION_H_
#define __NBODY_TENSOR_TENSOR_INDEX_EXPRESSION_H_

#include <algorithm>
#include <iterator>
#include <type_traits>

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
	std::size_t N, std::size_t M, std::size_t Dim, typename Scalar,
	bool Const, bool IsRef>
class TensorIndexReference final {
	
private:
	
	using Tensor = std::conditional_t<
		IsRef,
		std::conditional_t<
			Const,
			Tensor<N, M, Dim, Scalar> const&,
			Tensor<N, M, Dim, Scalar>&>
		Tensor<N, M, Dim, Scalar> >;
	
	Tensor _tensor;
	TensorIndex _upperIndices[N];
	TensorIndex _lowerIndices[M];
	
	TensorIndexExpression(
			Tensor tensor,
			TensorIndex upperIndices[N],
			TensorIndex lowerIndices[M]) :
			_tensor(tensor) {
		std::copy(
			std::begin(upperIndices),
			std::end(upperIndices),
			std::begin(_upperIndices));
		std::copy(
			std::begin(lowerIndices),
			std::end(lowerIndices),
			std::begin(_lowerIndices));
	}
	
public:
	
	operator TensorIndexReference<N, M, Dim, Scalar, true, IsRef>() const {
		return TensorIndexExpression<N, M, Dim, Scalar, true, Ref>(
			_tensor,
			_upperIndices,
			_lowerIndices);
	}
	
	template<typename Const_ = Const>
	std::enable_if_t<Const_, void> operator=(
			TensorIndexExpression<
				N, M, Dim, Scalar,
				Const, IsRef> const& rhs) const;
	
};

}

#endif

