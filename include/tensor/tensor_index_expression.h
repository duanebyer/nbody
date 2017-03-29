#ifndef __NBODY_TENSOR_TENSOR_INDEX_EXPRESSION_H_
#define __NBODY_TENSOR_TENSOR_INDEX_EXPRESSION_H_

#include <type_traits>

#include "tensor.h"

namespace nbody {

enum class TensorIndex {
	A, B, C, D, E, F, G,
	H, I, J, K, L, M, N, O, P,
	Q, R, S, T, U, V,
	W, X, Y, Z,
	// Now I know my A, B, Cs,...
};

template<
	std::size_t N, std::size_t M, std::size_t Dim, typename Scalar,
	std::size_t IndicesN, std::size_t IndicesM,
	bool Const>
class TensorIndexExpression final {
	
private:
	
	using TensorReference = std::conditional_t<
		Const,
		Tensor<N, M, Dim, Scalar> const&,
		Tensor<N, M, Dim, Scalar>&>;
	
	TensorReference _tensor;
	TensorIndex _upperIndices[IndicesN];
	TensorIndex _lowerIndices[IndicesM];
	
public:
	
	template<typename Const_ = Const>
	std::enable_if_t<Const_, void> operator=(
			TensorIndexExpression<
				N, M, Dim, Scalar,
				IndicesN, IndicesM> const& rhs);
	
};

}

#endif

