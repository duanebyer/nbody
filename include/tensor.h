#ifndef __NBODY_TENSOR_H_
#define __NBODY_TENSOR_H_

#include <cstddef>
#include <type_traits>

namespace nbody {

/**
 * \brief The floating point type used to represent scalars.
 */
using Scalar = double;

template<std::size_t N, std::size_t M, std::size_t Dim>
class Tensor;

/**
 * \brief A vector in a finite dimensional vector space.
 * 
 * A Vector can also be thought of as a (0, 1) Tensor.
 * 
 * \tparam Dim the dimension of the vector space
 */
template<std::size_t Dim>
using Vector = Tensor<0, 1, Dim>;

/**
 * \brief A covector in a finite dimensional dual space.
 * 
 * A Covector can be thought of as a linear mapping from a Vector to a Scalar,
 * or alternatively, as a (1, 0) Tensor.
 * 
 * \tparam Dim the dimension of the dual space
 */
template<std::size_t Dim>
using Covector = Tensor<1, 0, Dim>;

/**
 * \brief A rank (N, M) tensor in a finite dimensional space.
 * 
 * A rank (N, M) Tensor is defined to be a linear mapping of N Vector%s and M
 * Covector%s. It is represented by an array of Scalar%s, similar to a matrix
 * but in even higher dimensions.
 * 
 * \tparam N the number of contravariant (upper) indices
 * \tparam M the number of covariant (lower) indices
 * \tparam Dim the dimension of the underlying vector space
 */
template<std::size_t N, std::size_t M, std::size_t Dim>
class Tensor final {
	
private:
	
	using PrevTensor = typename std::conditional<N == 0,
	                                             Tensor<N, M - 1, Dim>,
	                                             Tensor<N - 1, M, Dim> >::type;
	
	PrevTensor _values[Dim];
	
public:
	
	/**
	 * \brief This type is a multidimensional array type capable of storing all
	 * of the coordinates of this Tensor.
	 */
	using Array = typename PrevTensor::Array[Dim];
	
	/**
	 * \brief Initializes a Tensor to the zero value.
	 */
	Tensor() {
	}
	
	/**
	 * \brief Initializes a Tensor with the specified coordinates.
	 * 
	 * \param values the coordinate values of the Tensor
	 */
	Tensor(Array values) {
		for (std::size_t index = 0; index < Dim; ++index) {
			PrevTensor tensor(values[index]);
			_values[index] = tensor;
		}
	}
	
	PrevTensor& operator[](std::size_t index) {
		return _values[index];
	}
	
	PrevTensor const& operator[](std::size_t index) const {
		return _values[index];
	}
	
	Tensor<N, M, Dim>& operator+=(Tensor<N, M, Dim> const& rhs) {
		for (std::size_t index = 0; index < Dim; ++index) {
			operator[](index) += rhs[index];
		}
		return *this;
	}
	
	Tensor<N, M, Dim>& operator-=(Tensor<N, M, Dim> const& rhs) {
		for (std::size_t index = 0; index < Dim; ++index) {
			operator[](index) -= rhs[index];
		}
		return *this;
	}
	
	Tensor<N, M, Dim>& operator*=(Scalar rhs) {
		for (std::size_t index = 0; index < Dim; ++index) {
			operator[](index) *= rhs;
		}
		return *this;
	}
	
	Tensor<N, M, Dim>& operator/=(Scalar rhs) {
		for (std::size_t index = 0; index < Dim; ++index) {
			operator[](index) /= rhs;
		}
		return *this;
	}
	
};

/**
 * \brief A rank zero tensor is the base case of the Tensor template.
 * 
 * This type can be used in any context where a Scalar could be used. Note
 * that two rank zero Tensor%s with different dimensions are not considered
 * the same type.
 */
template<std::size_t Dim>
class Tensor<0, 0, Dim> final {
	
private:
	
	Scalar _value;
	
public:
	
	using Array = Scalar;
	
	Tensor() : _value(0) {
	}
	
	Tensor(Scalar value): _value(value) {
	}
	
	operator Scalar&() {
		return _value;
	}
	
	operator Scalar() const {
		return _value;
	}
	
};

template<std::size_t N, std::size_t M, std::size_t Dim>
bool operator==(Tensor<N, M, Dim> const& lhs, Tensor<N, M, Dim> const& rhs) {
	bool result = true;
	for (std::size_t index = 0; index < Dim; ++index) {
		result = result && (lhs[index] == rhs[index]);
	}
	return result;
}

template<std::size_t N, std::size_t M, std::size_t Dim>
bool operator!=(Tensor<N, M, Dim> const& lhs, Tensor<N, M, Dim> const& rhs) {
	return !operator==(lhs, rhs);
}

template<std::size_t N, std::size_t M, std::size_t Dim>
Tensor<N, M, Dim> operator+(Tensor<N, M, Dim> lhs,
                            Tensor<N, M, Dim> const& rhs) {
	lhs += rhs;
	return lhs;
}

template<std::size_t N, std::size_t M, std::size_t Dim>
Tensor<N, M, Dim> operator-(Tensor<N, M, Dim> lhs,
                            Tensor<N, M, Dim> const& rhs) {
	lhs -= rhs;
	return lhs;
}

template<std::size_t N, std::size_t M, std::size_t Dim>
Tensor<N, M, Dim> operator*(Scalar lhs, Tensor<N, M, Dim> rhs) {
	rhs *= lhs;
	return rhs;
}

template<std::size_t N, std::size_t M, std::size_t Dim>
Tensor<N, M, Dim> operator*(Tensor<N, M, Dim> lhs, Scalar rhs) {
	lhs *= rhs;
	return lhs;
}

template<std::size_t N, std::size_t M, std::size_t Dim>
Tensor<N, M, Dim> operator/(Tensor<N, M, Dim> lhs, Scalar rhs) {
	lhs /= rhs;
	return lhs;
}

}

#endif

