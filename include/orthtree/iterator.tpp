#ifndef __NBODY_ORTHTREE_ITERATOR_TPP_
#define __NBODY_ORTHTREE_ITERATOR_TPP_

#include "orthtree/iterator.h"

#include "orthtree/reference.h"

namespace nbody {

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
template<bool Const>
struct Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafIteratorBase<Const>::
PointerProxy final {
	
	ReferenceProxy reference;
	
	PointerProxy(ReferenceProxy reference) :
			reference(reference) {
	}
	
	ReferenceProxy* operator->() {
		return &reference;
	}
	
};

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
template<bool Const>
Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafIteratorBase<Const>::reference
Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafIteratorBase<Const>::
operator*() const {
	return LeafReferenceProxyBase<Const>(
		_orthtree->_leafs[_index].position,
		_orthtree->_leafs[_index].value);
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
template<bool Const>
Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafIteratorBase<Const>::pointer
Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafIteratorBase<Const>::
operator->() const {
	return PointerProxy(operator*());
}



template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
template<bool Const>
struct Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIteratorBase<Const>::
PointerProxy final {
	
	ReferenceProxy reference;
	
	PointerProxy(ReferenceProxy reference) :
			reference(reference) {
	}
	
	ReferenceProxy* operator->() {
		return &reference;
	}
	
};

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
template<bool Const>
Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIteratorBase<Const>::reference
Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIteratorBase<Const>::
operator*() const {
	NodeIteratorBase<Const, false> parent(
		_orthtree,
		_index + _orthtree->_nodes[_index].parentIndex);
	NodeIteratorBase<Const, false> children[(1 << Dim) + 1];
	for (
			std::size_t childIndex = 0;
			childIndex < (1 << Dim) + 1;
			++childIndex) {
		children[childIndex] = NodeIteratorBase<Const, false>(
			_orthtree,
			_index + _orthtree->_nodes[_index].childIndices[childIndex]);
	}
	LeafRangeBase<Const> leafs(
		_orthtree,
		_orthtree->_nodes[_index].leafIndex,
		_orthtree->_nodes[_index].leafIndex +
		_orthtree->_nodes[_index].leafCount);
	return NodeReferenceProxyBase<Const>(
		parent,
		children,
		leafs,
		_orthtree->_nodes[_index].hasParent,
		_orthtree->_nodes[_index].hasChildren,
		_orthtree->_nodes[_index].depth,
		_orthtree->_nodes[_index].position,
		_orthtree->_nodes[_index].dimensions,
		_orthtree->_nodes[_index].value,
		std::make_index_sequence<(1 << Dim) + 1>());
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
template<bool Const>
Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIteratorBase<Const>::pointer
Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIteratorBase<Const>::
operator->() const {
	return PointerProxy(operator*());
}

}

#endif

