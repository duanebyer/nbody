#ifndef __NBODY_ORTHTREE_RANGE_H_
#define __NBODY_ORTHTREE_RANGE_H_

#include <type_traits>

#include "orthtree/orthtree.h"

#include "orthtree/iterator.h"

namespace nbody {

/**
 * \brief A pseudo-container that provides access to a collection of leaves
 * from an Orthtree.
 * 
 * The leaves are stored in depth-first order.
 * 
 * This container partially meets the requirements of `ReversibleContainer`. The
 * differences arise because a range cannot be created, and which elements it
 * contains cannot be changed. A range must be obtained from the Orthtree class.
 * 
 * \tparam Const whether this container allows for modifying its elements
 */
template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
template<bool Const>
class Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafRangeBase final {
	
private:
	
	friend Orthtree<Dim, Vector, LeafValue, NodeValue>;
	
	template<bool Const_, bool Reverse_>
	friend NodeIteratorBase;
	
	using OrthtreePointer = std::conditional_t<
			Const,
			Orthtree<Dim, Vector, LeafValue, NodeValue> const*,
			Orthtree<Dim, Vector, LeafValue, NodeValue>*>;
	
	OrthtreePointer _orthtree;
	typename LeafIterator::size_type _lowerIndex;
	typename LeafIterator::size_type _upperIndex;
	
	LeafRangeBase(
			OrthtreePointer orthtree,
			typename LeafIterator::size_type lowerIndex,
			typename LeafIterator::size_type upperIndex) :
			_orthtree(orthtree),
			_lowerIndex(lowerIndex),
			_upperIndex(upperIndex) {
	}
	
public:
	
	// Container typedefs.
	using iterator = LeafIterator;
	using const_iterator = ConstLeafIterator;
	using reverse_iterator = ReverseLeafIterator;
	using const_reverse_iterator = ConstReverseLeafIterator;
	
	using value_type = typename iterator::value_type;
	using reference = typename iterator::reference;
	using const_reference = typename const_iterator::reference;
	using pointer = typename iterator::pointer;
	using const_pointer = typename const_iterator::pointer;
	using size_type = typename iterator::size_type;
	using difference_type = typename iterator::difference_type;
	
	operator LeafRangeBase<true>() const {
		return LeafRangeBase<true>(_orthtree, _lowerIndex, _upperIndex);
	}
	
	// Container iteration range methods.
	std::conditional_t<Const, const_iterator, iterator> begin() const {
		return std::conditional_t<
			Const,
			ConstLeafIterator,
			LeafIterator>(
			_orthtree,
			_lowerIndex);
	}
	const_iterator cbegin() const {
		return ConstLeafIterator(_orthtree, _lowerIndex);
	}
	
	std::conditional_t<Const, const_iterator, iterator> end() const {
		return std::conditional_t<
			Const,
			ConstLeafIterator,
			LeafIterator>(
			_orthtree,
			_upperIndex);
	}
	const_iterator cend() const {
		return ConstLeafIterator(_orthtree, _upperIndex);
	}
	
	std::conditional_t<Const, const_reverse_iterator, reverse_iterator>
			rbegin() const {
		return std::conditional_t<
			Const,
			ConstReverseLeafIterator,
			ReverseLeafIterator>(
			_orthtree, _upperIndex - 1);
	}
	const_reverse_iterator crbegin() const {
		return ConstReverseLeafIterator(_orthtree, _upperIndex - 1);
	}
	
	std::conditional_t<Const, const_reverse_iterator, reverse_iterator>
			rend() const {
		return std::conditional_t<
			Const,
			ConstReverseLeafIterator,
			ReverseLeafIterator>(
			_orthtree, _lowerIndex - 1);
	}
	const_reverse_iterator crend() const {
		return ConstReverseLeafIterator(_orthtree, _lowerIndex - 1);
	}
	
	// Container size methods.
	size_type size() const {
		return _upperIndex - _lowerIndex;
	}
	size_type max_size() const {
		return size();
	}
	bool empty() const {
		return _upperIndex == _lowerIndex;
	}
	
};



/**
 * \brief A pseudo-container that provides access to a collection of nodes
 * from an Orthtree.
 * 
 * The nodes are stored in depth-first order.
 * 
 * This container partially meets the requirements of `ReversibleContainer`. The
 * differences arise because a range cannot be created, and which elements it
 * contains cannot be changed. A range must be obtained from the Orthtree class.
 * 
 * \tparam Const whether this container allows for modifying its elements
 */
template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
template<bool Const>
class Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeRangeBase final {
	
private:
	
	friend Orthtree;
	
	using OrthtreePointer = std::conditional_t<
			Const,
			Orthtree<L, N, Dim> const*,
			Orthtree<L, N, Dim>*>;
	
	OrthtreePointer _orthtree;
	typename NodeIterator::size_type _lowerIndex;
	typename NodeIterator::size_type _upperIndex;
	
	NodeRangeBase(
			OrthtreePointer orthtree,
			typename NodeIterator::size_type lowerIndex,
			typename NodeIterator::size_type upperIndex) :
			_orthtree(orthtree),
			_lowerIndex(lowerIndex),
			_upperIndex(upperIndex) {
	}
	
public:
	
	// Container typedefs.
	using iterator = NodeIterator;
	using const_iterator = ConstNodeIterator;
	using reverse_iterator = ReverseNodeIterator;
	using const_reverse_iterator = ConstReverseNodeIterator;
	
	using value_type = typename iterator::value_type;
	using reference = typename iterator::reference;
	using const_reference = typename const_iterator::reference;
	using pointer = typename iterator::pointer;
	using const_pointer = typename const_iterator::pointer;
	using size_type = typename iterator::size_type;
	using difference_type = typename iterator::difference_type;
	
	operator NodeRangeBase<true>() const {
		return NodeRangeBase<true>(_orthtree, _lowerIndex, _upperIndex);
	}
	
	// Container iteration range methods.
	std::conditional_t<Const, const_iterator, iterator> begin() const {
		return std::conditional_t<
			Const,
			ConstNodeIterator,
			NodeIterator>(
			_orthtree,
			_lowerIndex);
	}
	const_iterator cbegin() const {
		return ConstNodeIterator(_orthtree, _lowerIndex);
	}
	
	std::conditional_t<Const, const_iterator, iterator> end() const {
		return std::conditional_t<
			Const,
			ConstNodeIterator,
			NodeIterator>(
			_orthtree,
			_upperIndex);
	}
	const_iterator cend() const {
		return ConstNodeIterator(_orthtree, _upperIndex);
	}
	
	std::conditional_t<Const, const_reverse_iterator, reverse_iterator>
			rbegin() const {
		return std::conditional_t<
			Const,
			ConstReverseNodeIterator,
			ReverseNodeIterator>(
			_orthtree, _upperIndex - 1);
	}
	const_reverse_iterator crbegin() const {
		return ConstReverseNodeIterator(_orthtree, _upperIndex - 1);
	}
	
	std::conditional_t<Const, const_reverse_iterator, reverse_iterator>
			rend() const {
		return std::conditional_t<
			Const,
			ConstReverseNodeIterator,
			ReverseNodeIterator>(
			_orthtree, _lowerIndex - 1);
	}
	const_reverse_iterator crend() const {
		return ConstReverseNodeIterator(_orthtree, _lowerIndex - 1);
	}
	
	// Container size methods.
	size_type size() const {
		return _upperIndex - _lowerIndex;
	}
	size_type max_size() const {
		return size();
	}
	bool empty() const {
		return _upperIndex == _lowerIndex;
	}
	
};

}

#endif

