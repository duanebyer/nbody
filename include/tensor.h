#ifndef __NBODY_TENSOR_H_
#define __NBODY_TENSOR_H_

#include <cstddef>
#include <initializer_list>
#include <iterator>
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
 * Covector%s. It is represented by an array of Scalar coordinates, similar to
 * a matrix but in even higher dimensions. Some of the coordinate indices are
 * contravariant (written like \f$A^i\f$), and some of the indices are
 * covariant (written like \f$A_i\f$). Covariant and contravariant indices are
 * treated separately. When using this class, all contravariant indices are
 * written first and covariant indices follow after. For instance,
 * \f$A^ij_kl\f$ is written in code as `A[i][j][k][l]`.
 * 
 * \tparam N the number of contravariant (upper) indices
 * \tparam M the number of covariant (lower) indices
 * \tparam Dim the dimension of the underlying vector space
 */
template<std::size_t N, std::size_t M, std::size_t Dim>
class Tensor final {
	
private:
	
	using PrevTensor = std::conditional_t<N == 0,
	                                      Tensor<N, M - 1, Dim>,
	                                      Tensor<N - 1, M, Dim> >;
	
	PrevTensor _values[Dim];
	
public:
	
	/**
	 * \brief This type is a multidimensional array type capable of storing all
	 * of the coordinates of this Tensor.
	 */
	using Array = typename PrevTensor::Array[Dim];
	
	static const std::size_t size = Dim * PrevTensor::size;
	
	/**
	 * \brief Initializes a Tensor to the zero value.
	 */
	Tensor() {
	}
	
	///@{
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
	
	Tensor(std::initializer_list<Scalar> values) :
			Tensor(values.begin(), values.end()) {
	}
	///@}
	
	/**
	 * \brief Initializes a Tensor with the specified coordinates.
	 * 
	 * \tparam It any `RandomAccessIterator`
	 * \param begin the start of the range of values
	 * \param end the end of the range of values
	 */
	template<typename It>
	Tensor(It begin, It end) {
		for (std::size_t index = 0; index < Dim; ++index) {
			PrevTensor tensor(begin, end);
			_values[index] = tensor;
			if ((std::size_t) (end - begin) >= PrevTensor::size) {
				begin += PrevTensor::size;
			}
			else {
				begin = end;
			}
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
	
	static const std::size_t size = 1;
	
	Tensor() : _value(0) {
	}
	
	Tensor(Scalar value): _value(value) {
	}
	
	template<typename It>
	Tensor(It begin, It end) {
		_value = begin == end ? 0 : *begin;
	}
	
	operator Scalar&() {
		return _value;
	}
	
	operator Scalar() const {
		return _value;
	}
	
};

/**
 * \brief Returns the identity matrix.
 * 
 * The identity matrix is a (1, 1) Tensor that takes a Vector and a Covector
 * and returns the sum of the component-wise products.
 */
template<std::size_t Dim>
Tensor<1, 1, Dim> const& identity() {
	static Tensor<1, 1, Dim> identity;
	if (identity[0][0] == 0.0) {
		for (std::size_t index = 0; index < Dim; ++index) {
			identity[index][index] = 1.0;
		}
	}
	return identity;
}

template<std::size_t N, std::size_t M, std::size_t Dim>
bool operator==(Tensor<N, M, Dim> const& lhs, Tensor<N, M, Dim> const& rhs) {
	bool result = true;
	for (std::size_t index = 0; index < Dim; ++index) {
		result = result && (lhs[index] == rhs[index]);
	}
	return result;
}

template<std::size_t Dim>
bool operator==(Tensor<0, 0, Dim> const& lhs, Tensor<0, 0, Dim> const& rhs) {
	return (Scalar) lhs == (Scalar) rhs;
}

template<std::size_t N, std::size_t M, std::size_t Dim>
bool operator!=(Tensor<N, M, Dim> const& lhs, Tensor<N, M, Dim> const& rhs) {
	return !operator==(lhs, rhs);
}

/**
 * \brief Returns the tensor product of two Tensor%s.
 */
template<std::size_t N1, std::size_t M1,
         std::size_t N2, std::size_t M2,
         std::size_t Dim>
Tensor<N1 + N2, M1 + M2, Dim> operator*(Tensor<N1, M1, Dim> const& lhs,
                                        Tensor<N2, M2, Dim> const& rhs) {
	// First case: both tensors have contravariant indices, so we must take the
	// product over the left-hand-side contravariant indices first.
	Tensor<N1 + N2, M1 + M2, Dim> result;
	for (std::size_t index = 0; index < Dim; ++index) {
		result[index] = lhs[index] * rhs;
	}
	return result;
}

template<std::size_t M1,
         std::size_t N2, std::size_t M2,
         std::size_t Dim>
Tensor<N2, M1 + M2, Dim> operator*(Tensor<0, M1, Dim> const& lhs,
                                   Tensor<N2, M2, Dim> const& rhs) {
	// Second case: only the second tensor has contravariant indices. Since the
	// contravariant indices must come first in the resulting tensor, take the
	// product over the contravariant indices of the right-hand-side.
	Tensor<N2, M1 + M2, Dim> result;
	for (std::size_t index = 0; index < Dim; ++index) {
		result[index] = lhs * rhs[index];
	}
	return result;
}

template<std::size_t M1, std::size_t M2, std::size_t Dim>
Tensor<0, M1 + M2, Dim> operator*(Tensor<0, M1, Dim> const& lhs,
                                  Tensor<0, M2, Dim> const& rhs) {
	// Third case: neither tensor has contravariant indices, so the covariant
	// indices of the left-hand-side should come next.
	Tensor<0, M1 + M2, Dim> result;
	for (std::size_t index = 0; index < Dim; ++index) {
		result[index] = lhs[index] * rhs;
	}
	return result;
}

template<std::size_t M2, std::size_t Dim>
Tensor<0, M2, Dim> operator*(Tensor<0, 0, Dim> const& lhs,
                             Tensor<0, M2, Dim> const& rhs) {
	// Fourth case: the left-hand-side has no indices of any kind, meaning it
	// is essentially a scalar. Simply take the scalar product of it with the
	// right-hand-side. This is the base case.
	return ((Scalar) lhs) * rhs;
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

