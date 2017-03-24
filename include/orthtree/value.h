#ifndef __NBODY_ORTHTREE_VALUE_H_
#define __NBODY_ORTHTREE_VALUE_H_

#include "orthtree/orthtree.h"

#include "orthtree/iterator.h"

namespace nbody {

/**
 * \brief Provides access to a leaf of the Orthtree.
 * 
 * Note that this type is a proxy to the actual implementation type. Internally,
 * Orthtree uses a private type to store the leaf data.
 */
template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
struct Orthtree<Dim, Vector, LeafValue, NodeValue>::Leaf final {
	
	Vector const position;
	LeafValue value;
	
	Leaf(Vector position, LeafValue value) :
			position(position),
			value(value) {
	}
	
};



/**
 * \brief Provides access to a node of the Orthtree.
 * 
 * Note that this type is a proxy to the actual implementation type. Internally,
 * Orthtree uses a private type to store the node data.
 * 
 * This type comes in `const` and non-`const` variants, since it provides access
 * to traversal of the Orthtree. The `const` variant stores ConstNodeIterator%s
 * to the parent and children of this node, while the non-`const` variant stores
 * NodeIterator%s.
 * 
 * \tparam Const whether this is a `const` variant of the node
 */
template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
template<bool Const>
struct Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeBase final {
	
	NodeIteratorBase<Const, false> const parent;
	/**
	 * \brief An array of iterators, one to each child of the Node.
	 * 
	 * Note that this array has a length of 1 larger than the actual number of
	 * children. The last element of the array points to the next sibling of
	 * this Node.
	 */
	NodeIteratorBase<Const, false> const children[(1 << Dim) + 1];
	
	LeafRangeBase<Const> const leafs;
	
	bool const hasParent;
	bool const hasChildren;
	
	/**
	 * \brief The number of generations below the root node that this node is
	 * located.
	 * 
	 * The root node has a depth of 0, its children have a depth of 1, and so
	 * on.
	 */
	typename NodeList::size_type const depth;
	
	Vector const position;
	Vector const dimensions;
	
	NodeValue value;
	
	NodeBase(
			NodeIteratorBase<Const, false> parent,
			NodeIteratorBase<Const, false> children[(1 << Dim) + 1],
			LeafRangeBase<Const> leafs,
			bool hasParent,
			bool hasChildren,
			typename NodeList::size_type depth,
			Vector position,
			Vector dimensions,
			NodeValue value) :
			parent(parent),
			children(children),
			leafs(leafs),
			hasParent(hasParent),
			hasChildren(hasChildren),
			depth(depth),
			position(position),
			dimensions(dimensions),
			value(value) {
	}
	
};

}

#endif

