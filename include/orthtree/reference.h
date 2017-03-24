#ifndef __NBODY_ORTHTREE_REFERENCE_H_
#define __NBODY_ORTHTREE_REFERENCE_H_

#include <type_traits>

#include "orthtree/orthtree.h"

#include "orthtree/iterator.h"
#include "orthtree/value.h"

namespace nbody {

/**
 * \brief Proxy type that acts as a reference to a Leaf.
 * 
 * This type is able to mimic the behaviour of the `Leaf&` type in most cases.
 * However, pointers to this type do not behave correctly. For example, the
 * address-of operator does not work as it would for a true reference.
 * 
 * \tparam Const whether this is a `const` variant of the reference
 */
template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
template<bool Const>
struct Orthtree<Dim, Vector, LeafValue, NodeValue>::
LeafReferenceProxyBase final {
	
private:
	
	template<bool Const_, bool Reverse_>
	friend LeafIteratorBase;
	
	using LeafReference = std::conditional_t<
		Const,
		Leaf const&,
		Leaf&>;
	using ValueReference = std::conditional_t<
		Const,
		LeafValue const&,
		LeafValue>;
	
	LeafReferenceProxyBase(Vector const& position, ValueReference value) :
		position(position),
		value(value) {
	}
	
public:
	
	Vector const& position;
	ValueReference value;
	
	LeafReferenceProxyBase(LeafReference leaf) : LeafReferenceProxyBase(
			leaf.position,
			leaf.value) {
	}
	
	operator LeafReferenceProxyBase<true>() const {
		return LeafReferenceProxyBase<true>(position, value);
	}
	
	operator Leaf() const {
		return Leaf(position, value);
	}
	
};



/**
 * \brief Proxy type that acts as a reference to a Node.
 * 
 * This type is able to mimic the behaviour of the `Leaf&` type in most cases.
 * However, pointers to this type do not behave correctly. For example, the
 * address-of operator does not work as it would for a true reference.
 * 
 * \tparam Const whether this is a `const` variant of the reference
 */
template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
template<bool Const>
struct Orthtree<Dim, Vector, LeafValue, NodeValue>::
NodeReferenceProxyBase final {
	
private:
	
	template<bool Const_, bool Reverse_>
	friend NodeIteratorBase;
	
	using NodeReference = std::conditional_t<
		Const,
		Node const&,
		Node&>;
	using ValueReference = std::conditional_t<
		Const,
		NodeValue const&,
		NodeValue&>;
	
	template<std::size_t... Index>
	NodeReferenceProxyBase(
			NodeIteratorBase<Const, false> parent,
			NodeIteratorBase<Const, false> children[(1 << Dim) + 1],
			LeafRangeBase<Const> leafs,
			bool const& hasParent,
			bool const& hasChildren,
			typename NodeList::size_type const& depth,
			Vector const& position,
			Vector const& dimensions,
			ValueReference value,
			std::index_sequence<Index...>) :
			parent(parent),
			children{children[Index]...},
			leafs(leafs),
			hasParent(hasParent),
			hasChildren(hasChildren),
			depth(depth),
			position(position),
			dimensions(dimensions),
			value(value) {
	}
	
public:
	
	NodeIteratorBase<Const, false> const parent;
	NodeIteratorBase<Const, false> const children[(1 << Dim) + 1];
	
	LeafRangeBase<Const> const leafs;
	
	bool const& hasParent;
	bool const& hasChildren;
	
	typename NodeList::size_type const& depth;
	
	Vector const& position;
	Vector const& dimensions;
	
	ValueReference value;
	
	NodeReferenceProxyBase(NodeReference node) : NodeReferenceProxyBase(
			node.parent,
			node.children,
			node.leafs,
			node.hasParent,
			node.hasChildren,
			node.depth,
			node.position,
			node.dimensions,
			node.value,
			std::make_index_sequence<(1 << Dim) + 1>()) {
	}
	
	operator NodeReferenceProxyBase<true>() const {
		return NodeReferenceProxyBase<true>(position, value);
	}
	
	operator NodeBase<Const>() const {
		return NodeBase<Const>(
			parent,
			children,
			leafs,
			hasParent,
			hasChildren,
			depth,
			position,
			dimensions,
			value);
	}
	
};
}

#endif

