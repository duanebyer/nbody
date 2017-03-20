#ifndef __NBODY_OCTREE_H_
#define __NBODY_OCTREE_H_

#include <climits>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <vector>

#include "tensor.h"

namespace nbody {

/**
 * \brief A data structure that stores spatial data in arbitrary dimensional
 * space.
 * 
 * This class is a fairly standard implementation of an octree that stores data
 * at discrete points. In addition, this class allows for data to be stored at
 * the nodes of the Octree (for example, the center of mass of all of the data
 * points contained within a node).
 * 
 * This class can be thought of as a pair of containers: one container of leaf
 * data (see the LeafViewBase class), and another container of nodes (see the
 * NodeViewBase class). These containers can be accessed using the Octree:leafs
 * and Octree::nodes methods. Traversing the structure of the Octree can be done
 * with the NodeIteratorBase and LeafIteratorBase classes.
 * 
 * Data can be added to or removed from the Octree through the Octree::insert,
 * Octree::erase, and Octree::move methods.
 * 
 * \tparam L the type of data stored at the leaves of the Octree
 * \tparam N the type of data stored at the nodes of the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename L, typename N, std::size_t Dim>
class Octree final {
	
public:
	
	static_assert(
		std::is_default_constructible<N>::value,
		"type N must be default constructible");
	static_assert(
		std::is_copy_assignable<N>::value,
		"type N must be copy assignable");
	static_assert(
		std::is_copy_constructible<N>::value,
		"type N must be copy constructible");
	static_assert(
		std::is_copy_assignable<L>::value,
		"type L must be copy assignable");
	static_assert(
		std::is_copy_constructible<L>::value,
		"type L must be copy constructible");
	
	// These classes are used as proxies for accessing the nodes and leafs of
	// the octree.
	
	struct Leaf;
	
	template<bool Const>
	struct LeafReferenceProxyBase;
	
	template<bool Const>
	struct NodeBase;
	
	template<bool Const>
	struct NodeReferenceProxyBase;
	
	using Node = NodeBase<false>;
	using ConstNode = NodeBase<true>;
	
	// TODO: Doxygen comment here
	using LeafReferenceProxy = LeafReferenceProxyBase<false>;
	using ConstLeafReferenceProxy = LeafReferenceProxyBase<true>;
	
	using NodeReferenceProxy = NodeReferenceProxyBase<false>;
	using ConstNodeReferenceProxy = NodeReferenceProxyBase<true>;
	
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
	 * \brief Pseudo-container that provides access to the leaves of the Octree
	 * without allowing for insertion or deletion of elements.
	 */
	using LeafRange = LeafRangeBase<false>;
	using ConstLeafRange = LeafRangeBase<true>;
	///@}
	
	///@{
	/**
	 * \brief Depth-first iterator over all of the leaves contained in the
	 * Octree.
	 */
	using LeafIterator = LeafIteratorBase<false, false>;
	using ConstLeafIterator = LeafIteratorBase<true, false>;
	using ReverseLeafIterator = LeafIteratorBase<false, true>;
	using ConstReverseLeafIterator = LeafIteratorBase<true, true>;
	///@}
	
	///@{
	/**
	 * \brief Pseudo-container that provides access to the nodes of the Octree
	 * without allowing for insertion or deletion of elements.
	 */
	using NodeRange = NodeRangeBase<false>;
	using ConstNodeRange = NodeRangeBase<true>;
	///@}
	
	///@{
	/**
	 * \brief Depth-first iterator over all of the nodes contained in the
	 * Octree.
	 */
	using NodeIterator = NodeIteratorBase<false, false>;
	using ConstNodeIterator = NodeIteratorBase<true, false>;
	using ReverseNodeIterator = NodeIteratorBase<false, true>;
	using ConstReverseNodeIterator = NodeIteratorBase<true, true>;
	///@}
	
private:
	
	struct LeafInternal;
	struct NodeInternal;
	
	using LeafList = std::vector<LeafInternal>;
	using NodeList = std::vector<NodeInternal>;
	
	// This class packages leaf data with a position.
	struct LeafInternal final {
		
		L value;
		Vector<Dim> position;
		
		Leaf(L value, Vector<Dim> position) :
				value(value),
				position(position) {
		}
		
	};
	
	// This class packages node data within the node heirarchy.
	struct NodeInternal final {
		
		// The depth of this node within the octree (0 for root, and so on).
		NodeList::size_type depth;
		
		// Whether this node has any children.
		bool hasChildren;
		// The indices of the children of this node, stored relative to the
		// index of this node. The last entry points to the next sibling of this
		// node, and is used to determine the total size of all of this node's
		// children.
		NodeList::size_type childIndices[(1 << Dim) + 1];
		
		// Whether this node has a parent.
		bool hasParent;
		// The relative index of the parent of this node.
		NodeList::difference_type parentIndex;
		// Which child # of its parent this node is. (0th child, 1st child,
		// etc).
		NodeList::size_type siblingIndex;
		
		// The number of leaves that this node contains. This includes leaves
		// stored by all descendants of this node.
		LeafList::size_type leafCount;
		// The index within the octree's leaf array that this node's leaves are
		// located at.
		LeafList::size_type leafIndex;
		
		// The section of space that this node encompasses.
		Vector<Dim> position;
		Vector<Dim> dimensions;
		
		// The data stored at the node itself.
		N value;
		
		// By default, a node is constructed as if it were an empty root node.
		Node() :
				depth(0),
				hasChildren(false),
				childIndices(),
				hasParent(false),
				parentIndex(),
				siblingIndex(),
				leafCount(0),
				leafIndex(0),
				position(),
				dimensions(),
				value() {
		}
		
	};
	
	// A list storing all of the leafs of the octree.
	LeafList _leafs;
	
	// A list storing all of the nodes of the octree.
	NodeList _nodes;
	
	// The number of leaves to store at a single node of the octree.
	LeafList::size_type _nodeCapacity;
	
	// The maximum depth of the octree.
	NodeList::size_type _maxDepth;
	
	// Whether the tree should be automatically readjust itself so that each
	// node has less leaves than the node capacity, as well as having as few
	// children as possible. If this is false, then the adjust() method has to
	// be called to force an adjustment.
	bool _adjust;
	
	
	
	// Determines whether a node can store a certain number of additional (or
	// fewer) leafs.
	bool canHoldLeafs(
			ConstNodeIterator node,
			LeafList::difference_type n) const {
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
			L const& leaf,
			Vector<Dim> const& position);
	
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
	 * \brief Constructs a new, empty Octree.
	 * 
	 * \param position { the location of the "upper-left" corner of the region
	 * of space that the Octree covers }
	 * \param dimensions the size of the region of space that the Octree covers
	 * \param nodeCapacity { the number of leaves that can be stored at
	 * one node }
	 * \param maxDepth the maximum number of generations of nodes
	 * \param adjust { whether the Octree should automatically create and
	 * destroy nodes to optimize the number of leaves per node }
	 */
	Octree(
			Vector<Dim> position,
			Vector<Dim> dimensions,
			LeafList::size_type nodeCapacity = 1,
			NodeList::size_type maxDepth = sizeof(Scalar) * CHAR_BIT,
			bool adjust = true);
	
	LeafList::size_type nodeCapacity() const {
		return _nodeCapacity;
	}
	NodeList::size_type maxDepth() const {
		return _maxDepth;
	}
	bool adjust() const {
		return _adjust;
	}
	
	///@{
	/**
	 * \brief Gets a pseudo-container that contains the leaves of the Octree.
	 * 
	 * The leaves are returned in depth-first order. See LeafRangeBase for
	 * the API of the resulting container.
	 */
	LeafRange leafs();
	ConstLeafRange leafs() const;
	ConstLeafRange cleafs() const;
	///@}
	
	///@{
	/**
	 * \brief Gets a pseudo-container that contains the nodes of the Octree.
	 * 
	 * The nodes are returned in depth-first order, with the root node as the
	 * first element in the container. See NodeRangeBase for the API of the
	 * resulting container.
	 */
	NodeRange nodes();
	ConstNodeRange nodes() const;
	ConstNodeRange cnodes() const;
	///@}
	
	// TODO: doxygen comment
	NodeIterator root();
	ConstNodeIterator root() const;
	ConstNodeRange croot() const;
	
	// TODO: doxygen comment
	NodeRange descendants(NodeIterator node);
	ConstNodeRange descendants(ConstNodeIterator node) const;
	ConstNodeRange cdescendants(ConstNodeIterator node) const;
	
	///@{
	/**
	 * \brief Creates and destroys nodes to optimize the number of leaves stored
	 * at each node.
	 * 
	 * This method will check for nodes that contain more than the maximum
	 * number of leaves, as well as for unnecessary nodes. The node structure
	 * of the Octree will be adjusted so that these situations are resolved.
	 * 
	 * If the Octree was constructed to automatically adjust itself, then this
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
		return adjust(nodes().begin());
	}
	///@}
	
	///@{
	/**
	 * \brief Adds a new leaf to the Octree.
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
			L const& data,
			Vector<Dim> const& position);
	
	std::tuple<NodeIterator, LeafIterator> insert(
			L const& data,
			Vector<Dim> const& position);
	///@}
	
	///@{
	/**
	 * \brief Adds a new leaf to the Octree.
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
			LeafTuple leafPair);
	
	template<typename LeafTuple>
	std::tuple<NodeIterator, LeafIterator> insert(
			ConstNodeIterator hint,
			LeafTuple leafPair);
	///@}
	
	///@{
	/**
	 * \brief Removes an leaf from the Octree.
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
			LeafIterator leaf);
	///@}
	
	///@{
	/**
	 * \brief Changes the position of a leaf within the Octree.
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
			Vector<Dim> const& position);
	
	std::tuple<NodeIterator, NodeIterator, LeafIterator> move(
			LeafIterator leaf,
			Vector<Dim> const& position);
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
			Vector<Dim> const& position);
	
	NodeIterator find(
			Vector<Dim> const& position) {
		return find(nodes().begin(), position);
	}
	
	ConstNodeIterator find(
			ConstNodeIterator start,
			Vector<Dim> const& position) const {
		return const_cast<Octree<L, N, Dim>*>(this)->find(start, position);
	}
	
	ConstNodeIterator find(
			Vector<Dim> const& position) const {
		return find(nodes().begin(), position);
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
		return const_cast<Octree<L, N, Dim>*>(this)->find(hint, leaf);
	}
	
	ConstNodeIterator find(
			ConstLeafIterator leaf) const {
		return find(nodes().begin(), leaf);
	}
	///@}
	
	// TODO: add doxygen comment
	NodeIterator findChild(
			ConstNodeIterator node,
			Vector<Dim> const& point);
	ConstNodeIterator findChild(
			ConstNodeIterator node,
			Vector<Dim> const& point) const {
		return const_cast<Octree<L, N, Dim>*>(this)->findChild(node, point);
	}
	
	// TODO: add doxygen comment
	NodeIterator findChild(
			ConstNodeIterator node,
			ConstLeafIterator leaf);
	ConstNodeIterator findChild(
			ConstNodeIterator node,
			ConstLeafIterator leaf) {
		return const_cast<Octree<L, N, Dim>*>(this)->findChild(node, leaf);
	}
	
	// TODO: add Doxygen comment here
	bool contains(ConstNodeIterator node, Vector<Dim> const& point) const {
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (!(
					point[dim] >= node->position[dim] &&
					point[dim] < node->position[dim] + node->dimensions[dim])) {
				return false;
			}
		}
		return true;
	}
	
	// TODO: add Doxygen comment here
	bool contains(ConstNodeIterator node, ConstLeafIterator leaf) const {
		LeafList::difference_type index = node.internalIt()->leafIndex;
		LeafList::size_type count = node.internalIt()->leafCount;
		return
			leaf._index >= index &&
			leaf._index < index + count;
	}
	
};



// TODO: Doxygen comment block
template<typename L, typename N, std::size_t Dim>
struct Octree<L, N, Dim>::Leaf final {
	
	Vector<Dim> const position;
	L value;
	
	Leaf(Vector<Dim> position, L value) :
			position(position),
			value(value) {
	}
	
};



// TODO: Doxygen comment block
template<typename L, typename N, std::size_t Dim>
template<bool Const>
struct Octree<L, N, Dim>::LeafReferenceProxyBase final {
	
private:
	
	template<bool Const_, bool Reverse_>
	friend class Octree<L, N, Dim>::LeafIteratorBase;
	
	using LeafReference = std::conditional_t<
		Const,
		Leaf const&,
		Leaf&>;
	using ValueReference = std::conditional_t<
		Const,
		L const&,
		L>;
	
	LeafReferenceProxyBase(Vector<Dim> const& position, ValueReference value) :
		position(position),
		value(value) {
	}
	
public:
	
	Vector<Dim> const& position;
	ValueReference value;
	
	LeafReferenceProxyBase(LeafReference leaf) :
			position(leaf.position),
			value(leaf.value) {
	}
	
	operator LeafReferenceProxyBase<true>() const {
		return LeafReferenceProxyBase<true>(position, value);
	}
	
	operator Leaf() const {
		return Leaf(position, value);
	}
	
};



/**
 * \brief An iterator over the LeafRangeBase container.
 * 
 * The leaves are iterated over in depth-first order.
 * 
 * This iterator meets the requirements of `RandomAccessIterator`. It also
 * contains some simple functions for querying the properties of a leaf of an
 * Octree.
 * 
 * \tparam Const whether this is a `const` variant of the iterator
 * \tparam Reverse whether this is a reverse or forward iterator
 */
template<typename L, typename N, std::size_t Dim>
template<bool Const, bool Reverse>
class Octree<L, N, Dim>::LeafIteratorBase final {
	
private:
	
	friend Octree<L, N, Dim>;
	
	template<bool Const_>
	friend class Octree<L, N, Dim>::LeafRangeBase;
	
	using ReferenceProxy = LeafReferenceProxyBase<Const>;
	using ListIterator = std::conditional_t<
			Const,
			LeafList::const_iterator,
			LeafList::iterator>;
	using ListReference = std::conditional_t<
			Const,
			LeafList::const_reference,
			LeafList::reference>;
	using OctreePointer = std::conditional_t<
			Const,
			Octree<L, N, Dim> const*,
			Octree<L, N, Dim>*>;
	
	struct PointerProxy {
		
		ReferenceProxy reference;
		
		PointerProxy(ReferenceProxy reference) :
				reference(reference) {
		}
		
		ReferenceProxy* operator->() const {
			return &reference;
		}
		
	};
	
	OctreePointer _octree;
	List::difference_type _index;
	
	LeafIteratorBase(OctreePointer octree, List::difference_type index) :
			_octree(octree),
			_index(index) {
	}
	
	// Converts this iterator to an iterator over the internal leaf type.
	ListIterator internalIt() const {
		return _octree->_leafs.begin() + _index;
	}
	
public:
	
	// Iterator typedefs.
	using value_type = Leaf;
	using reference = ReferenceProxy;
	using pointer = PointerProxy;
	using size_type = List::size_type;
	using difference_type = List::difference_type;
	using iterator_category = std::input_iterator_tag;
	
	LeafIteratorBase() :
			_octree(NULL),
			_index(0) {
	}
	
	operator LeafIteratorBase<true, Reverse>() const {
		return LeafIteratorBase<true, Reverse>(_octree, _index);
	}
	
	/**
	 * \brief Flips the direction of the iterator.
	 * 
	 * This method transforms a forward iterator into a reverse iterator, and a
	 * reverse iterator into a forward iterator. Note that the resulting
	 * does not point to the same value as the original iterator (same behaviour
	 * as `std::reverse_iterator<...>::base()`).
	 */
	LeafIteratorBase<Const, !Reverse> reverse() const {
		difference_type shift = !Reverse ? -1 : +1;
		return LeafIteratorBase<Const, !Reverse>(_octree, _index + shift);
	}
	
	// Iterator element access methods.
	reference operator*() const {
		return LeafReferenceProxyBase<Const>(
			_octree->_leafs[_index].position,
			_octree->_leafs[_index].value);
	}
	pointer operator->() const {
		return PointerProxy(operator*());
	}
	
	// Iterator increment methods.
	LeafIteratorBase<Const, Reverse>& operator++() {
		difference_type shift = Reverse ? -1 : +1;
		_index += shift;
		return *this;
	}
	LeafIteratorBase<Const, Reverse> operator++(int) {
		LeafIteratorBase<Const, Reverse> result = *this;
		operator++();
		return result;
	}
	
	LeafIteratorBase<Const, Reverse>& operator--() {
		difference_type shift = Reverse ? +1 : -1;
		_index += shift;
		return *this;
	}
	LeafIteratorBase<Const, Reverse> operator--(int) {
		LeafIteratorBase<Const, Reverse> result = *this;
		operator--();
		return result;
	}
	
	// Iterator comparison methods.
	friend bool operator==(
			LeafIteratorBase<Const, Reverse> const& lhs,
			LeafIteratorBase<Const, Reverse> const& rhs) {
		return lhs._octree == rhs._octree && lhs._index == rhs._index;
	}
	friend bool operator!=(
			LeafIteratorBase<Const, Reverse> const& lhs,
			LeafIteratorBase<Const, Reverse> const& rhs) {
		return !(lhs == rhs);
	}
	
};



/**
 * \brief A pseudo-container that provides access to a collection of leaves
 * from an Octree.
 * 
 * The leaves are stored in depth-first order.
 * 
 * This container partially meets the requirements of `SequenceContainer` and
 * `ReversibleContainer`. The only differences in behaviour arise because
 * elements cannot be added to or removed from this container. In addition, this
 * container cannot be created, but must be retrieved using the Octree class.
 * 
 * \tparam Const whether this container allows for modifying its elements
 */
template<typename L, typename N, std::size_t Dim>
template<bool Const>
class Octree<L, N, Dim>::LeafRangeBase final {
	
private:
	
	friend Octree<L, N, Dim>;
	
	template<bool Const_, bool Reverse_>
	friend class Octree<L, N, Dim>::NodeIteratorBase;
	
	using OctreePointer = std::conditional_t<
			Const,
			Octree<L, N, Dim> const*,
			Octree<L, N, Dim>*>;
	
	OctreePointer _octree;
	LeafIterator::size_type _lowerIndex;
	LeafIterator::size_type _upperIndex;
	
	LeafRangeBase(
			OctreePointer octree,
			LeafIterator::size_type lowerIndex,
			LeafIterator::size_type upperIndex) :
			_octree(octree),
			_lowerIndex(lowerIndex),
			_upperIndex(upperIndex) {
	}
	
public:
	
	// Container typedefs.
	using iterator = LeafIterator;
	using const_iterator = ConstLeafIterator;
	using reverse_iterator = ReverseLeafIterator;
	using const_reverse_iterator = ConstReverseLeafIterator;
	
	using value_type = iterator::value_type;
	using reference = iterator::reference;
	using const_reference = const_iterator::reference;
	using pointer = iterator::pointer;
	using const_pointer = const_iterator::pointer;
	using size_type = iterator::size_type;
	using difference_type = iterator::difference_type;
	
	operator LeafRangeBase<true>() const {
		return LeafRangeBase<true>(_octree, _lowerIndex, _upperIndex);
	}
	
	// Container iteration range methods.
	std::conditional_t<Const, const_iterator, iterator> begin() const {
		return std::conditional_t<
			Const,
			const_iterator,
			iterator>(
			_octree,
			_lowerIndex);
	}
	const_iterator cbegin() const {
		return const_iterator(_octree, _lowerIndex);
	}
	
	std::conditional_t<Const, const_iterator, iterator> end() const {
		return std::conditional_t<
			Const,
			const_iterator,
			iterator>(
			_octree,
			_upperIndex);
	}
	const_iterator cend() const {
		return const_iterator(_octree, _upperIndex);
	}
	
	std::conditional_t<Const, const_reverse_iterator, reverse_iterator>
			rbegin() const {
		return std::conditional_t<
			Const,
			const_reverse_iterator,
			reverse_iterator>(
			_octree, _upperIndex - 1);
	}
	const_reverse_iterator crbegin() const {
		return const_reverse_iterator(_octree, _upperIndex - 1);
	}
	
	std::conditional_t<Const, const_reverse_iterator, reverse_iterator>
			rend() const {
		return std::conditional_t<
			Const,
			const_reverse_iterator,
			reverse_iterator>(
			_octree, _lowerIndex - 1);
	}
	const_reverse_iterator crend() const {
		return const_reverse_iterator(_octree, _lowerIndex - 1);
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
	
	// Container element access methods.
	std::conditional_t<Const, const_reference, reference> front() const {
		return _octree->_leafs[_lowerIndex].data;
	}
	std::conditional_t<Const, const_reference, reference> back() const {
		return _octree->_leafs[_upperIndex - 1].data;
	}
	
	std::conditional_t<Const, const_reference, reference> operator[](
			size_type n) const {
		return _octree->_leafs[_lowerIndex + n].data;
	}
	std::conditional_t<Const, const_reference, reference> at(
			size_type n) const {
		return _octree->_leafs.at(_lowerIndex + n).data;
	}
	
};



// TODO: Doxygen comment here
template<typename L, typename N, std::size_t Dim>
template<bool Const>
struct Octree<L, N, Dim>::Node final {
	
	NodeIteratorBase<Const> const parent;
	NodeIteratorBase<Const> const children[(1 << Dim) + 1];
	
	LeafRangeBase<Const> const leafs;
	
	bool const hasParent;
	bool const hasChildren;
	
	NodeList::size_type depth;
	
	Vector<Dim> const position;
	Vector<Dim> const dimensions;
	
	N value;
	
	Node(
			NodeIteratorBase<Const> parent,
			NodeIteratorBase<Const>[(1 << Dim) + 1] children,
			LeafRangeBase<Const> leafs,
			bool hasChildren,
			Vector<Dim> position,
			Vector<Dim> dimensions,
			N value) :
			parent(parent),
			children(children),
			leafs(leafs),
			hasChildren(hasChildren),
			position(position),
			dimensions(dimensions),
			value(value) {
	}
	
};



// TODO: Doxygen comment here
template<typename L, typename N, std::size_t Dim>
template<Const>
struct Octree<L, N, Dim>::NodeReferenceProxyBase final {
	
private:
	
	template<bool Const_, bool Reverse_>
	friend class Octree<L, N, Dim>::NodeIteratorBase;
	
	using NodeReference = std::conditional_t<
		Const,
		Node const&,
		Node&>;
	using ValueReference = std::conditional_t<
		Const,
		N const&,
		N>;
	
	NodeReferenceProxyBase(
		NodeIteratorBase<Const, false> parent,
		NodeIteratorBase<Const, false> children[(1 << Dim) + 1],
		LeafRangeBase<Const, false> leafs,
		bool const& hasParent,
		bool const& hasChildren,
		std::size_t depth,
		Vector<Dim> const& position,
		Vector<Dim> const& dimensions,
		ValueReference value) :
		parent(parent),
		children(children),
		leafs(leafs),
		hasChildren(hasChildren),
		position(position),
		dimensions(dimensions),
		value(value) {
	}
	
public:
	
	NodeIteratorBase<Const, false> const parent;
	NodeIteratorBase<Const, false> const children[(1 << Dim) + 1];
	
	LeafRangeBase<Const, false> const leafs;
	
	bool const& hasParent;
	bool const& hasChildren;
	
	std::size_t const& depth;
	
	Vector<Dim> const& position;
	Vector<Dim> const& dimensions;
	
	ValueReference value;
	
	NodeReferenceProxyBase(NodeReference node) :
			parent(node.parent),
			children(node.children),
			leafs(node.leafs),
			hasParent(node.hasParent),
			hasChildren(node.hasChildren),
			depth(node.depth),
			position(node.position),
			dimensions(node.dimensions),
			value(node.value) {
	}
	
	operator NodeReferenceProxyBase<true>() const {
		return NodeReferenceProxyBase<true>(position, value);
	}
	
	operator Node<Const>() const {
		return Node(
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



// TODO: this needs to be ripped out and remade.
/**
 * \brief An iterator over the NodeRangeBase container.
 * 
 * The nodes are iterated over in depth-first order.
 * 
 * This iterator meets the requirements of `RandomAccessIterator`. It also
 * contains some simple functions for querying the properties of a node of an
 * Octree.
 * 
 * \tparam Const whether this is a `const` variant of the iterator
 * \tparam Reverse whether this is a reverse or forward iterator
 */
template<typename L, typename N, std::size_t Dim>
template<bool Const, bool Reverse>
class Octree<L, N, Dim>::NodeIteratorBase final {
	
private:
	
	friend Octree<L, N, Dim>;
	
	template<bool Const_>
	friend class Octree<L, N, Dim>::NodeRangeBase;
	
	using ReferenceProxy = NodeReferenceProxyBase<Const>;
	using ListIterator = std::conditional_t<
			Const,
			NodeList::const_iterator,
			NodeList::iterator>;
	using ListReference = std::conditional_t<
			Const,
			NodeList::const_reference,
			NodeList::reference>;
	using OctreePointer = std::conditional_t<
			Const,
			Octree<L, N, Dim> const*,
			Octree<L, N, Dim>*>;
	
	struct PointerProxy {
		
		ReferenceProxy reference;
		
		PointerProxy(ReferenceProxy reference) :
				reference(reference) {
		}
		
		ReferenceProxy* operator->() const {
			return &reference;
		}
		
	};
	
	OctreePointer _octree;
	List::difference_type _index;
	
	NodeIteratorBase(OctreePointer octree, List::difference_type index) :
			_octree(octree),
			_index(index) {
	}
	
	// Converts this iterator to an iterator over the internal node type.
	ListIterator internalIt() const {
		return _octree->_nodes.begin() + _index;
	}
	
public:
	
	// Iterator typedefs.
	using value_type = Node;
	using reference = ReferenceProxy;
	using pointer = PointerProxy;
	using size_type = List::size_type;
	using difference_type = List::difference_type;
	using iterator_category = std::input_iterator_tag;
	
	NodeIteratorBase() :
			_octree(NULL),
			_index(0) {
	}
	
	operator NodeIteratorBase<true, Reverse>() const {
		return NodeIteratorBase<true, Reverse>(_octree, _index);
	}
	
	/**
	 * \brief Flips the direction of the iterator.
	 * 
	 * This method transforms a forward iterator into a reverse iterator, and a
	 * reverse iterator into a forward iterator. Note that the resulting
	 * does not point to the same value as the original iterator (same behaviour
	 * as `std::reverse_iterator<...>::base()`).
	 */
	NodeIteratorBase<Const, !Reverse> reverse() const {
		difference_type shift = !Reverse ? -1 : +1;
		return NodeIteratorBase<Const, !Reverse>(_octree, _index + shift);
	}
	
	// Iterator element access methods.
	reference operator*() const {
		NodeIteratorBase<Const, false> parent(
			_octree,
			_index + _octree->_nodes[_index].parentIndex);
		NodeIteratorBase<Const, false> children[(1 << Dim) + 1];
		for (
				std::size_t childIndex = 0;
				childIndex < (1 << Dim) + 1;
				++childIndex) {
			children[childIndex] = NodeIteratorBase<Const, false>(
				_octree,
				_index + _octree->_nodes[_index].childIndices[childIndex]);
		}
		LeafRangeBase<Const> leafs(
			_octree,
			_octree->_nodes[_index].leafIndex,
			_octree->_nodes[_index].leafIndex +
			_octree->_nodes[_index].leafCount);
		return NodeReferenceProxyBase<Const>(
			parent,
			children,
			leafs,
			_octree->_nodes[_index].hasParent,
			_octree->_nodes[_index].hasChildren,
			_octree->_nodes[_index].depth,
			_octree->_nodes[_index].position,
			_octree->_nodes[_index].dimensions,
			_octree->_nodes[_index].value);
	}
	pointer operator->() const {
		return PointerProxy(operator*());
	}
	
	// Iterator increment methods.
	NodeIteratorBase<Const, Reverse>& operator++() {
		difference_type shift = Reverse ? -1 : +1;
		_index += shift;
		return *this;
	}
	NodeIteratorBase<Const, Reverse> operator++(int) {
		NodeIteratorBase<Const, Reverse> result = *this;
		operator++();
		return result;
	}
	
	NodeIteratorBase<Const, Reverse>& operator--() {
		difference_type shift = Reverse ? +1 : -1;
		_index += shift;
		return *this;
	}
	NodeIteratorBase<Const, Reverse> operator--(int) {
		NodeIteratorBase<Const, Reverse> result = *this;
		operator--();
		return result;
	}
	
	// Iterator comparison methods.
	friend bool operator==(
			NodeIteratorBase<Const, Reverse> const& lhs,
			NodeIteratorBase<Const, Reverse> const& rhs) {
		return lhs._octree == rhs._octree && lhs._index == rhs._index;
	}
	friend bool operator!=(
			NodeIteratorBase<Const, Reverse> const& lhs,
			NodeIteratorBase<Const, Reverse> const& rhs) {
		return !(lhs == rhs);
	}
	
};



/**
 * \brief A pseudo-container that provides access to a collection of nodes
 * from an Octree.
 * 
 * The nodes are stored in depth-first order.
 * 
 * This container partially meets the requirements of `SequenceContainer` and
 * `ReversibleContainer`. The only differences in behaviour arise because
 * elements cannot be added to or removed from this container. In addition, this
 * container cannot be created, but must be retrieved using the Octree class.
 * 
 * \tparam Const whether this container allows for modifying its elements
 */
template<typename L, typename N, std::size_t Dim>
template<bool Const>
class Octree<L, N, Dim>::NodeRangeBase final {
	
private:
	
	friend Octree<L, N, Dim>;
	
	using OctreePointer = std::conditional_t<
			Const,
			Octree<L, N, Dim> const*,
			Octree<L, N, Dim>*>;
	
	OctreePointer _octree;
	std::size_t _lowerIndex;
	std::size_t _upperIndex;
	
	NodeRangeBase(
			OctreePointer octree,
			std::size_t lowerIndex,
			std::size_t upperIndex) :
			_octree(octree),
			_lowerIndex(lowerIndex),
			_upperIndex(upperIndex) {
	}
	
public:
	
	// Container typedefs.
	using value_type = N;
	using reference = N&;
	using const_reference = N const&;
	using pointer = N*;
	using const_pointer = N const*;
	using iterator =
			Octree<L, N, Dim>::NodeIterator;
	using const_iterator =
			Octree<L, N, Dim>::ConstNodeIterator;
	using reverse_iterator =
			Octree<L, N, Dim>::ReverseNodeIterator;
	using const_reverse_iterator =
			Octree<L, N, Dim>::ConstReverseNodeIterator;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	
	operator NodeRangeBase<true>() const {
		return NodeRangeBase<true>(_octree, _lowerIndex, _upperIndex);
	}
	
	// Container iteration range methods.
	std::conditional_t<Const, const_iterator, iterator> begin() const {
		return std::conditional_t<
			Const,
			const_iterator,
			iterator>(
			_octree,
			_lowerIndex);
	}
	const_iterator cbegin() const {
		return const_iterator(_octree, _lowerIndex);
	}
	
	std::conditional_t<Const, const_iterator, iterator> end() const {
		return std::conditional_t<
			Const,
			const_iterator,
			iterator>(
			_octree,
			_upperIndex);
	}
	const_iterator cend() const {
		return const_iterator(_octree, _upperIndex);
	}
	
	std::conditional_t<Const, const_reverse_iterator, reverse_iterator>
			rbegin() const {
		return std::conditional_t<
			Const,
			const_reverse_iterator,
			reverse_iterator>(
			_octree,
			_upperIndex - 1);
	}
	const_reverse_iterator crbegin() const {
		return const_reverse_iterator(_octree, _upperIndex - 1);
	}
	
	std::conditional_t<Const, const_reverse_iterator, reverse_iterator>
			rend() const {
		return std::conditional_t<
			Const,
			const_reverse_iterator,
			reverse_iterator>(
			_octree,
			_lowerIndex - 1);
	}
	const_reverse_iterator crend() const {
		return const_reverse_iterator(_octree, _lowerIndex - 1);
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
	
	// Container element access methods.
	std::conditional_t<Const, const_reference, reference> front() const {
		return _octree->_nodes[_lowerIndex].data;
	}
	std::conditional_t<Const, const_reference, reference> back() const {
		return _octree->_nodes[_upperIndex - 1].data;
	}
	
	std::conditional_t<Const, const_reference, reference> operator[](
			size_type n) const {
		return _octree->_nodes[_lowerIndex + n].data;
	}
	std::conditional_t<Const, const_reference, reference> at(
			size_type n) const {
		return _octree->_nodes.at(_lowerIndex + n).data;
	}
	
};

}

#include "octree.tpp"

#endif

