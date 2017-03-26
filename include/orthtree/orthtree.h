#ifndef __NBODY_ORTHTREE_ORTHTREE_H_
#define __NBODY_ORTHTREE_ORTHTREE_H_

#include <array>
#include <climits>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "internal/functional.h"
#include "internal/type_traits.h"

namespace nbody {

/**
 * \brief A data structure that stores spatial data in arbitrary dimensional
 * space.
 * 
 * An orthtree is the extension of a quadtree/octree to arbitrary dimensional
 * space. This class implements an orthtree that stores data at discrete points
 * (the leaves) as well as at the nodes of the underlying tree structure.
 * 
 * The Orthtree class is designed to be compatible with the STL container
 * library. It acts as two different kinds of containers: one container of leaf
 * data (see the LeafRangeBase class), and another container of nodes (see the
 * NodeRangeBase class). These containers can be accessed using the
 * Orthtree::leafs and Orthtree::nodes methods.
 * 
 * Data can be added to or removed from the Orthtree through the
 * Orthtree::insert, Orthtree::erase, and Orthtree::move methods.
 * 
 * \tparam Dim the dimension of the space that the Orthtree is embedded in
 * \tparam Vector a `Dim`-dimensional vector type that supports `operator[]`
 * \tparam LeafValue the type of data stored at the leaves of the Orthtree
 * \tparam NodeValue the type of data stored at the nodes of the Orthtree
 */
template<
	std::size_t Dim,
	typename Vector = std::array<double, Dim>,
	typename LeafValue = std::tuple<>,
	typename NodeValue = std::tuple<> >
class Orthtree final {
	
public:
	
	// Make sure that the template parameters meet all of the conditions.
	
	using Scalar = std::remove_reference_t<decltype(std::declval<Vector>()[0])>;
	
	static_assert(
		Dim > 0,
		"template parameter Dim must be larger than 0");
	
	static_assert(
		std::is_default_constructible<NodeValue>::value,
		"template parameter NodeValue must be default constructible");
	static_assert(
		std::is_copy_assignable<NodeValue>::value,
		"template parameter NodeValue must be copy assignable");
	static_assert(
		std::is_copy_constructible<NodeValue>::value,
		"template parameter NodeValue must be copy constructible");
	
	static_assert(
		std::is_copy_assignable<LeafValue>::value,
		"template parameter LeafValue must be copy assignable");
	static_assert(
		std::is_copy_constructible<LeafValue>::value,
		"template parameter LeafValue must be copy constructible");
	
	static_assert(
		std::is_copy_constructible<Vector>::value,
		"template parameter Vector must be copy constructible");
	static_assert(
		nbody::internal::is_invocable<
			nbody::internal::subscript<Vector>,
			Vector,
			std::size_t>::value,
		"template parameter Vector must have operator[]");
	static_assert(
		nbody::internal::is_invocable_r<
			Scalar&, nbody::internal::subscript<Vector>,
			Vector,
			std::size_t>::value,
		"template parameter Vector must have operator[]");
	
	static_assert(
		nbody::internal::is_invocable_r<
			Scalar, std::plus<Scalar>, Scalar, Scalar>::value,
		"template parameter Vector's scalar type must have operator+");
	static_assert(
		nbody::internal::is_invocable_r<
			Scalar, std::minus<Scalar>, Scalar, Scalar>::value,
		"template parameter Vector's scalar type must have operator-");
	static_assert(
		nbody::internal::is_invocable_r<
			Scalar, std::multiplies<Scalar>, Scalar, Scalar>::value,
		"template parameter Vector's scalar type must have operator*");
	static_assert(
		nbody::internal::is_invocable_r<
			Scalar, std::multiplies<Scalar>, Scalar, Scalar>::value,
		"template parameter Vector's scalar type must have operator/");
	static_assert(
		nbody::internal::is_invocable_r<
			Scalar, std::less<Scalar>, Scalar, Scalar>::value,
		"template parameter Vector's scalar type must have operator<");
	static_assert(
		nbody::internal::is_invocable_r<
			Scalar, std::greater_equal<Scalar>, Scalar, Scalar>::value,
		"template parameter Vector's scalar type must have operator>=");
	
	// These classes are used as proxies for accessing the nodes and leafs of
	// the orthtree.
	
	struct Leaf;
	
	template<bool Const>
	struct LeafReferenceProxyBase;
	
	template<bool Const>
	struct NodeBase;
	
	template<bool Const>
	struct NodeReferenceProxyBase;
	
	///@{
	/**
	 * \brief Provides access to a node of the Orthtree.
	 * 
	 * These types are `const` and non-`const` specializations of the NodeBase
	 * class, since NodeBase allows for traversal of the Orthtree through
	 * iterators. Otherwise, it might be possible to obtain a non-`const`
	 * reference from a `const` iterator.
	 */
	using Node = NodeBase<false>;
	using ConstNode = NodeBase<true>;
	///@}
	
	///@{
	/**
	 * \brief Proxy type that acts as a reference to a Leaf.
	 * 
	 * \see LeafReferenceProxyBase
	 */
	using LeafReferenceProxy = LeafReferenceProxyBase<false>;
	using ConstLeafReferenceProxy = LeafReferenceProxyBase<true>;
	///@}
	
	///@{
	/**
	 * \brief Proxy type that acts as a reference to a Node.
	 * 
	 * \see NodeReferenceProxyBase
	 */
	using NodeReferenceProxy = NodeReferenceProxyBase<false>;
	using ConstNodeReferenceProxy = NodeReferenceProxyBase<true>;
	///@}
	
	// These classes are used to construct const-variants and reverse-variants
	// of the public range and iterator classes.
	
	template<bool Const, bool Reverse>
	class LeafIteratorBase;	
	
	template<bool Const>
	class LeafRangeBase;
	
	template<bool Const, bool Reverse>
	class NodeIteratorBase;	
	
	template<bool Const>
	class NodeRangeBase;
	
	///@{
	/**
	 * \brief Depth-first iterator over all of the leaves contained in the
	 * Orthtree.
	 * 
	 * \see LeafIteratorBase
	 */
	using LeafIterator = LeafIteratorBase<false, false>;
	using ConstLeafIterator = LeafIteratorBase<true, false>;
	using ReverseLeafIterator = LeafIteratorBase<false, true>;
	using ConstReverseLeafIterator = LeafIteratorBase<true, true>;
	///@}
	
	///@{
	/**
	 * \brief Pseudo-container that provides access to the leaves of the
	 * Orthtree without allowing for insertion or deletion of elements.
	 * 
	 * \see LeafRangeBase
	 */
	using LeafRange = LeafRangeBase<false>;
	using ConstLeafRange = LeafRangeBase<true>;
	///@}
	
	///@{
	/**
	 * \brief Depth-first iterator over all of the nodes contained in the
	 * Orthtree.
	 * 
	 * \see NodeIteratorBase
	 */
	using NodeIterator = NodeIteratorBase<false, false>;
	using ConstNodeIterator = NodeIteratorBase<true, false>;
	using ReverseNodeIterator = NodeIteratorBase<false, true>;
	using ConstReverseNodeIterator = NodeIteratorBase<true, true>;
	///@}
	
	///@{
	/**
	 * \brief Pseudo-container that provides access to the nodes of the Orthtree
	 * without allowing for insertion or deletion of elements.
	 * 
	 * \see NodeRangeBase
	 */
	using NodeRange = NodeRangeBase<false>;
	using ConstNodeRange = NodeRangeBase<true>;
	///@}
	
private:
	
	struct LeafInternal;
	struct NodeInternal;
	
	using LeafList = std::vector<LeafInternal>;
	using NodeList = std::vector<NodeInternal>;
	
	// This class packages leaf data with a position.
	struct LeafInternal final {
		
		LeafValue value;
		Vector position;
		
		LeafInternal(LeafValue value, Vector position) :
				value(value),
				position(position) {
		}
		
	};
	
	// This class packages node data within the node heirarchy.
	struct NodeInternal final {
		
		// The depth of this node within the orthtree (0 for root, and so on).
		typename NodeList::size_type depth;
		
		// Whether this node has any children.
		bool hasChildren;
		// The indices of the children of this node, stored relative to the
		// index of this node. The last entry points to the next sibling of this
		// node, and is used to determine the total size of all of this node's
		// children.
		typename NodeList::size_type childIndices[(1 << Dim) + 1];
		
		// Whether this node has a parent.
		bool hasParent;
		// The relative index of the parent of this node.
		typename NodeList::difference_type parentIndex;
		// Which child # of its parent this node is. (0th child, 1st child,
		// etc).
		typename NodeList::size_type siblingIndex;
		
		// The number of leaves that this node contains. This includes leaves
		// stored by all descendants of this node.
		typename LeafList::size_type leafCount;
		// The index within the orthtree's leaf array that this node's leaves
		// are located at.
		typename LeafList::size_type leafIndex;
		
		// The section of space that this node encompasses.
		Vector position;
		Vector dimensions;
		
		// The data stored at the node itself.
		NodeValue value;
		
		// By default, a node is constructed as if it were an empty root node.
		NodeInternal(Vector position, Vector dimensions) :
				depth(0),
				hasChildren(false),
				childIndices(),
				hasParent(false),
				parentIndex(),
				siblingIndex(),
				leafCount(0),
				leafIndex(0),
				position(position),
				dimensions(dimensions),
				value() {
		}
		
	};
	
	// A list storing all of the leafs of the orthtree.
	LeafList _leafs;
	
	// A list storing all of the nodes of the orthtree.
	NodeList _nodes;
	
	// The number of leaves to store at a single node of the orthtree.
	typename LeafList::size_type _nodeCapacity;
	
	// The maximum depth of the orthtree.
	typename NodeList::size_type _maxDepth;
	
	// Whether the tree should be automatically readjust itself so that each
	// node has less leaves than the node capacity, as well as having as few
	// children as possible. If this is false, then the adjust() method has to
	// be called to force an adjustment.
	bool _adjust;
	
	
	
	// Determines whether a node can store a certain number of additional (or
	// fewer) leafs.
	bool canHoldLeafs(
			ConstNodeIterator node,
			typename LeafList::difference_type n) const {
		return
			node->leafs.size() + n <= _nodeCapacity ||
			node->depth >= _maxDepth;
	}
	
	// Divides a node into a set of subnodes and partitions its leaves between
	// them. This function may reorganize the leaf vector (some leaf iterators
	// may become invalid).
	NodeIterator createChildren(NodeIterator node);
	
	// Destroys all descendants of a node and takes their leaves into the node.
	// This function will not reorganize the leaf vector (all leaf iterators
	// will remain valid).
	NodeIterator destroyChildren(NodeIterator node);
	
	// Adds a leaf to a specific node.
	LeafIterator insertAt(
			NodeIterator node,
			LeafValue const& value,
			Vector const& position);
	
	// Removes a leaf from a node.
	LeafIterator eraseAt(
			NodeIterator node,
			LeafIterator leaf);
	
	// Moves a leaf from this node to another one.
	LeafIterator moveAt(
			NodeIterator sourceNode,
			NodeIterator destNode,
			LeafIterator sourceLeaf);
	
public:
	
	/**
	 * \brief Constructs a new, empty Orthtree.
	 * 
	 * \param position { the location of the "upper-left" corner of the region
	 * of space that the Orthtree covers }
	 * \param dimensions the size of the region of space that the Orthtree
	 * covers
	 * \param nodeCapacity { the number of leaves that can be stored at
	 * one node }
	 * \param maxDepth the maximum number of generations of nodes
	 * \param adjust { whether the Orthtree should automatically create and
	 * destroy nodes to optimize the number of leaves per node }
	 */
	Orthtree(
			Vector position,
			Vector dimensions,
			typename LeafList::size_type nodeCapacity = 1,
			typename NodeList::size_type maxDepth = sizeof(Scalar) * CHAR_BIT,
			bool adjust = true);
	
	typename LeafList::size_type nodeCapacity() const {
		return _nodeCapacity;
	}
	typename NodeList::size_type maxDepth() const {
		return _maxDepth;
	}
	bool adjust() const {
		return _adjust;
	}
	
	///@{
	/**
	 * \brief Gets a range that contains the leaves of the Orthtree.
	 */
	LeafRange leafs();
	ConstLeafRange leafs() const;
	ConstLeafRange cleafs() const;
	///@}
	
	///@{
	/**
	 * \brief Gets a range that contains the nodes of the Orthtree.
	 */
	NodeRange nodes();
	ConstNodeRange nodes() const;
	ConstNodeRange cnodes() const;
	///@}
	
	///@{
	/**
	 * \brief Gets an iterator to the root node of the Orthtree.
	 */
	NodeIterator root();
	ConstNodeIterator root() const;
	ConstNodeIterator croot() const;
	///@}
	
	///@{
	/**
	 * \brief Gets a range that contains all descendants of a specific node.
	 */
	NodeRange descendants(ConstNodeIterator node);
	ConstNodeRange descendants(ConstNodeIterator node) const;
	ConstNodeRange cdescendants(ConstNodeIterator node) const;
	///@}
	
	///@{
	/**
	 * \brief Creates and destroys nodes to optimize the number of leaves stored
	 * at each node.
	 * 
	 * This method will check for nodes that contain more than the maximum
	 * number of leaves, as well as for unnecessary nodes. The node structure
	 * of the Orthtree will be adjusted so that these situations are resolved.
	 * 
	 * If the Orthtree was constructed to automatically adjust itself, then this
	 * method will never do anything.
	 * 
	 * NodeIterator%s may be invalidated.
	 * 
	 * \param node an iterator to the node which will be adjusted
	 * 
	 * \return whether any changes were actually made
	 */
	bool adjust(ConstNodeIterator node);
	bool adjust() {
		return adjust(root());
	}
	///@}
	
	///@{
	/**
	 * \brief Adds a new leaf to the Orthtree.
	 * 
	 * If the optional `hint` parameter is provided, then this method will begin
	 * its search for the node to insert the leaf at the `hint` node.
	 * 
	 * NodeIterator%s and LeafIterator%s may be invalidated.
	 * 
	 * \param hint a starting guess as to where the leaf should be placed
	 * \param data the data to be inserted at the leaf
	 * \param position the position of the leaf
	 * 
	 * \return a tuple containing the NodeIterator to the node that the
	 * leaf was added to, and a LeafIterator to the new leaf
	 */
	std::tuple<NodeIterator, LeafIterator> insert(
			ConstNodeIterator hint,
			LeafValue const& value,
			Vector const& position);
	
	std::tuple<NodeIterator, LeafIterator> insert(
			LeafValue const& value,
			Vector const& position) {
		return insert(root(), value, position);
	}
	///@}
	
	///@{
	/**
	 * \brief Adds a new leaf to the Orthtree.
	 * 
	 * If the optional `hint` parameter is provided, then this method will begin
	 * its search for the node to insert the leaf at the `hint` node.
	 * 
	 * The provided data should be packaged in a tuple. It will be accessed
	 * the `std::get` method.
	 * 
	 * NodeIterator%s and LeafIterator%s may be invalidated.
	 * 
	 * \param hint a starting guess as to where the leaf should be placed
	 * \param leafPair a tuple containing both the leaf and its position
	 * 
	 * \return a tuple containing the NodeIterator to the node that the
	 * leaf was added to, and a LeafIterator to the new leaf
	 */
	template<typename LeafTuple>
	std::tuple<NodeIterator, LeafIterator> insert(
			ConstNodeIterator hint,
			LeafTuple leafPair) {
		return insert(
			hint,
			std::get<LeafValue>(leafPair),
			std::get<Vector>(leafPair));
	}
	
	template<typename LeafTuple>
	std::tuple<NodeIterator, LeafIterator> insert(
			LeafTuple leafPair) {
		return insert(root(), leafPair);
	}
	///@}
	
	///@{
	/**
	 * \brief Removes an leaf from the Orthtree.
	 * 
	 * If the optional `hint` parameter is provided, then this method will begin
	 * its search for the node to remove the leaf from at the `hint` node.
	 * 
	 * NodeIterator%s and LeafIterator%s may be invalidated.
	 * 
	 * \param hint a starting guess as to where the leaf should be removed from
	 * \param leaf a LeafIterator to the leaf that should be removed
	 * 
	 * \return a tuple containing the NodeIterator that the leaf was removed
	 * from, and the LeafIterator to the leaf after the removed leaf
	 */
	std::tuple<NodeIterator, LeafIterator> erase(
			ConstNodeIterator hint,
			LeafIterator leaf);
	
	std::tuple<NodeIterator, LeafIterator> erase(
			LeafIterator leaf) {
		return erase(root(), leaf);
	}
	///@}
	
	///@{
	/**
	 * \brief Changes the position of a leaf within the Orthtree.
	 * 
	 * If the optional `hint` parameter is provided, then this method will begin
	 * its search for the node to move the leaf from at the `hint` node.
	 * 
	 * NodeIterator%s and LeafIterator%s may be invalidated.
	 * 
	 * \param hint a starting guess as to where the leaf should be moved from
	 * \param leaf a LeafIterator to the leaf that should be moved
	 * \param position the new position that the leaf should be moved to
	 * 
	 * \return a tuple containing the NodeIterator that the leaf was removed
	 * from, the NodeIterator that it was moved to, and the LeafIterator itself
	 */
	std::tuple<NodeIterator, NodeIterator, LeafIterator> move(
			ConstNodeIterator hint,
			LeafIterator leaf,
			Vector const& position);
	
	std::tuple<NodeIterator, NodeIterator, LeafIterator> move(
			LeafIterator leaf,
			Vector const& position) {
		return move(root(), leaf, position);
	}
	///@}
	
	///@{
	/**
	 * \brief Searches for the node that contains a certain position.
	 * 
	 * This method searches for the lowest-level node that contains a position.
	 * 
	 * If the optional `hint` parameter is provided, then this method will begin
	 * its search for the node at the `hint` node.
	 * 
	 * \param hint an initial guess as to what node contains the position
	 * \param position the position to search for
	 * 
	 * \return the node that contains the position
	 */
	NodeIterator find(
			ConstNodeIterator hint,
			Vector const& point);
	
	NodeIterator find(
			Vector const& point) {
		return find(root(), point);
	}
	
	ConstNodeIterator find(
			ConstNodeIterator start,
			Vector const& point) const {
		return const_cast<std::remove_const_t<decltype(*this)*> >(this)->
			find(start, point);
	}
	
	ConstNodeIterator find(
			Vector const& point) const {
		return find(root(), point);
	}
	///@}
	
	///@{
	/**
	 * \brief Searchs for the node that contains a certain leaf.
	 * 
	 * This method searches for the lowest-level node that contains a leaf.
	 * 
	 * If the optional `hint` parameter is provided, then this method will begin
	 * its search for the node at the `hint` node.
	 * 
	 * \param start an initial guess as to what Node contains the position
	 * \param leaf the leaf to search for
	 * 
	 * \return the node that contains the leaf
	 */
	NodeIterator find(
			ConstNodeIterator hint,
			ConstLeafIterator leaf);
	
	NodeIterator find(
			ConstLeafIterator leaf) {
		return find(nodes().begin(), leaf);
	}
	
	ConstNodeIterator find(
			ConstNodeIterator hint,
			ConstLeafIterator leaf) const {
		return const_cast<std::remove_const_t<decltype(*this)*> >(this)->
			find(hint, leaf);
	}
	
	ConstNodeIterator find(
			ConstLeafIterator leaf) const {
		return find(nodes().begin(), leaf);
	}
	///@}
	
	///@{
	/**
	 * \brief Searches a node for one of its children that contains a certain
	 * position.
	 * 
	 * This method divides all of space into the octants of the node. Even if
	 * the position is not technically contained by one of the children, the
	 * child that would contain the position if it was extended to infinity will
	 * returned.
	 * 
	 * If the node has no children, then the result is undefined.
	 * 
	 * \param node the parent of the children that will be searched
	 * \param position the position to search for
	 */
	NodeIterator findChild(
			ConstNodeIterator node,
			Vector const& point);
	ConstNodeIterator findChild(
			ConstNodeIterator node,
			Vector const& point) const {
		return const_cast<std::remove_const_t<decltype(*this)*> >(this)->
			findChild(node, point);
	}
	///@}
	
	///@{
	/**
	 * \brief Searches a node for one of its children that contains a certain
	 * leaf.
	 * 
	 * If the node has no children, then the result is undefined.
	 * 
	 * \param node the parent of the children that will be searched
	 * \param leaf the leaf to search for
	 */
	NodeIterator findChild(
			ConstNodeIterator node,
			ConstLeafIterator leaf);
	ConstNodeIterator findChild(
			ConstNodeIterator node,
			ConstLeafIterator leaf) const {
		return const_cast<std::remove_const_t<decltype(*this)*> >(this)->
			findChild(node, leaf);
	}
	///@}
	
	/**
	 * \brief Determines whether a node contains a point.
	 */
	bool contains(ConstNodeIterator node, Vector const& point) const;
	
	/**
	 * \brief Determines whether a node contains a leaf.
	 */
	bool contains(ConstNodeIterator node, ConstLeafIterator leaf) const;
	
};

}

#endif

