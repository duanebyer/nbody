#ifndef __NBODY_ORTHTREE_H_
#define __NBODY_ORTHTREE_H_

#include <climits>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensor.h"

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
 * \tparam L the type of data stored at the leaves of the Orthtree
 * \tparam N the type of data stored at the nodes of the Orthtree
 * \tparam Dim the dimension of the space that the Orthtree is embedded in
 */
template<typename L, typename N, std::size_t Dim>
class Orthtree final {
	
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
		
		L value;
		Vector<Dim> position;
		
		LeafInternal(L value, Vector<Dim> position) :
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
		Vector<Dim> position;
		Vector<Dim> dimensions;
		
		// The data stored at the node itself.
		N value;
		
		// By default, a node is constructed as if it were an empty root node.
		NodeInternal() :
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
			Vector<Dim> position,
			Vector<Dim> dimensions,
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
		return adjust(nodes().begin());
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
			L const& value,
			Vector<Dim> const& position);
	
	std::tuple<NodeIterator, LeafIterator> insert(
			L const& value,
			Vector<Dim> const& position);
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
			LeafTuple leafPair);
	
	template<typename LeafTuple>
	std::tuple<NodeIterator, LeafIterator> insert(
			ConstNodeIterator hint,
			LeafTuple leafPair);
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
			LeafIterator leaf);
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
			Vector<Dim> const& point);
	
	NodeIterator find(
			Vector<Dim> const& point) {
		return find(nodes().begin(), point);
	}
	
	ConstNodeIterator find(
			ConstNodeIterator start,
			Vector<Dim> const& point) const {
		return const_cast<Orthtree<L, N, Dim>*>(this)->find(start, point);
	}
	
	ConstNodeIterator find(
			Vector<Dim> const& point) const {
		return find(nodes().begin(), point);
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
		return const_cast<Orthtree<L, N, Dim>*>(this)->find(hint, leaf);
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
			Vector<Dim> const& point);
	ConstNodeIterator findChild(
			ConstNodeIterator node,
			Vector<Dim> const& point) const {
		return const_cast<Orthtree<L, N, Dim>*>(this)->findChild(node, point);
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
		return const_cast<Orthtree<L, N, Dim>*>(this)->findChild(node, leaf);
	}
	///@}
	
	/**
	 * \brief Determines whether a node contains a point.
	 */
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
	
	/**
	 * \brief Determines whether a node contains a leaf.
	 */
	bool contains(ConstNodeIterator node, ConstLeafIterator leaf) const {
		typename LeafList::difference_type index =
			(typename LeafList::difference_type) node.internalIt()->leafIndex;
		typename LeafList::difference_type count =
			(typename LeafList::difference_type) node.internalIt()->leafCount;
		return
			leaf._index >= index &&
			leaf._index < index + count;
	}
	
};



/**
 * \brief Provides access to a leaf of the Orthtree.
 * 
 * Note that this type is a proxy to the actual implementation type. Internally,
 * Orthtree uses a private type to store the leaf data.
 */
template<typename L, typename N, std::size_t Dim>
struct Orthtree<L, N, Dim>::Leaf final {
	
	Vector<Dim> const position;
	L value;
	
	Leaf(Vector<Dim> position, L value) :
			position(position),
			value(value) {
	}
	
};



/**
 * \brief Proxy type that acts as a reference to a Leaf.
 * 
 * This type is able to mimic the behaviour of the `Leaf&` type in most
 * cases. However, pointers to this type do not behave correctly. For
 * example, the address-of operator does not work as it would for a true
 * reference.
 * 
 * \tparam Const whether this is a `const` variant of the reference
 */
template<typename L, typename N, std::size_t Dim>
template<bool Const>
struct Orthtree<L, N, Dim>::LeafReferenceProxyBase final {
	
private:
	
	template<bool Const_, bool Reverse_>
	friend class Orthtree<L, N, Dim>::LeafIteratorBase;
	
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
 * \brief An iterator over the LeafRangeBase container.
 * 
 * The Leaf%s are iterated over in depth-first order.
 * 
 * This iterator meets the requirements of `InputIterator`. It would also
 * the requirements of `BidirectionalIterator` except that the reference type
 * used by this iterator is not `Leaf&`. A proxy reference type
 * LeafReferenceProxyBase is used instead.
 * 
 * \tparam Const whether this is a `const` variant of the iterator
 * \tparam Reverse whether this is a reverse or forward iterator
 */
template<typename L, typename N, std::size_t Dim>
template<bool Const, bool Reverse>
class Orthtree<L, N, Dim>::LeafIteratorBase final {
	
private:
	
	friend Orthtree<L, N, Dim>;
	
	template<bool Const_>
	friend class Orthtree<L, N, Dim>::LeafRangeBase;
	
	using ReferenceProxy = LeafReferenceProxyBase<Const>;
	using List = LeafList;
	using ListIterator = std::conditional_t<
			Const,
			typename LeafList::const_iterator,
			typename LeafList::iterator>;
	using ListReference = std::conditional_t<
			Const,
			typename LeafList::const_reference,
			typename LeafList::reference>;
	using OrthtreePointer = std::conditional_t<
			Const,
			Orthtree<L, N, Dim> const*,
			Orthtree<L, N, Dim>*>;
	
	// This type is used by the arrow (->) operator.
	struct PointerProxy {
		
		ReferenceProxy reference;
		
		PointerProxy(ReferenceProxy reference) :
				reference(reference) {
		}
		
		ReferenceProxy* operator->() {
			return &reference;
		}
		
	};
	
	OrthtreePointer _orthtree;
	typename List::difference_type _index;
	
	LeafIteratorBase(
			OrthtreePointer orthtree,
			typename List::difference_type index) :
			_orthtree(orthtree),
			_index(index) {
	}
	
	// Converts this iterator to an iterator over the internal leaf type.
	ListIterator internalIt() const {
		return _orthtree->_leafs.begin() + _index;
	}
	
public:
	
	// Iterator typedefs.
	using value_type = Leaf;
	using reference = ReferenceProxy;
	using pointer = PointerProxy;
	using size_type = typename List::size_type;
	using difference_type = typename List::difference_type;
	using iterator_category = std::input_iterator_tag;
	
	LeafIteratorBase() :
			_orthtree(NULL),
			_index(0) {
	}
	
	operator LeafIteratorBase<true, Reverse>() const {
		return LeafIteratorBase<true, Reverse>(_orthtree, _index);
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
		return LeafIteratorBase<Const, !Reverse>(_orthtree, _index + shift);
	}
	
	// Iterator element access methods.
	reference operator*() const {
		return LeafReferenceProxyBase<Const>(
			_orthtree->_leafs[_index].position,
			_orthtree->_leafs[_index].value);
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
		return lhs._orthtree == rhs._orthtree && lhs._index == rhs._index;
	}
	friend bool operator!=(
			LeafIteratorBase<Const, Reverse> const& lhs,
			LeafIteratorBase<Const, Reverse> const& rhs) {
		return !(lhs == rhs);
	}
	
};



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
template<typename L, typename N, std::size_t Dim>
template<bool Const>
class Orthtree<L, N, Dim>::LeafRangeBase final {
	
private:
	
	friend Orthtree<L, N, Dim>;
	
	template<bool Const_, bool Reverse_>
	friend class Orthtree<L, N, Dim>::NodeIteratorBase;
	
	using OrthtreePointer = std::conditional_t<
			Const,
			Orthtree<L, N, Dim> const*,
			Orthtree<L, N, Dim>*>;
	
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
			const_iterator,
			iterator>(
			_orthtree,
			_lowerIndex);
	}
	const_iterator cbegin() const {
		return const_iterator(_orthtree, _lowerIndex);
	}
	
	std::conditional_t<Const, const_iterator, iterator> end() const {
		return std::conditional_t<
			Const,
			const_iterator,
			iterator>(
			_orthtree,
			_upperIndex);
	}
	const_iterator cend() const {
		return const_iterator(_orthtree, _upperIndex);
	}
	
	std::conditional_t<Const, const_reverse_iterator, reverse_iterator>
			rbegin() const {
		return std::conditional_t<
			Const,
			const_reverse_iterator,
			reverse_iterator>(
			_orthtree, _upperIndex - 1);
	}
	const_reverse_iterator crbegin() const {
		return const_reverse_iterator(_orthtree, _upperIndex - 1);
	}
	
	std::conditional_t<Const, const_reverse_iterator, reverse_iterator>
			rend() const {
		return std::conditional_t<
			Const,
			const_reverse_iterator,
			reverse_iterator>(
			_orthtree, _lowerIndex - 1);
	}
	const_reverse_iterator crend() const {
		return const_reverse_iterator(_orthtree, _lowerIndex - 1);
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
template<typename L, typename N, std::size_t Dim>
template<bool Const>
struct Orthtree<L, N, Dim>::NodeBase final {
	
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
	
	Vector<Dim> const position;
	Vector<Dim> const dimensions;
	
	N value;
	
	NodeBase(
			NodeIteratorBase<Const, false> parent,
			NodeIteratorBase<Const, false> children[(1 << Dim) + 1],
			LeafRangeBase<Const> leafs,
			bool hasParent,
			bool hasChildren,
			typename NodeList::size_type depth,
			Vector<Dim> position,
			Vector<Dim> dimensions,
			N value) :
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



/**
 * \brief Proxy type that acts as a reference to a Node.
 * 
 * This type is able to mimic the behaviour of the `Node&` type in most
 * cases. However, pointers to this type do not behave correctly. For
 * example, the address-of operator does not work as it would for a true
 * reference.
 * 
 * \tparam Const whether this is a `const` variant of the reference
 */
template<typename L, typename N, std::size_t Dim>
template<bool Const>
struct Orthtree<L, N, Dim>::NodeReferenceProxyBase final {
	
private:
	
	template<bool Const_, bool Reverse_>
	friend class Orthtree<L, N, Dim>::NodeIteratorBase;
	
	using NodeReference = std::conditional_t<
		Const,
		Node const&,
		Node&>;
	using ValueReference = std::conditional_t<
		Const,
		N const&,
		N&>;
	
	template<std::size_t... Index>
	NodeReferenceProxyBase(
			NodeIteratorBase<Const, false> parent,
			NodeIteratorBase<Const, false> children[(1 << Dim) + 1],
			LeafRangeBase<Const> leafs,
			bool const& hasParent,
			bool const& hasChildren,
			typename NodeList::size_type const& depth,
			Vector<Dim> const& position,
			Vector<Dim> const& dimensions,
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
	
	Vector<Dim> const& position;
	Vector<Dim> const& dimensions;
	
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



/**
 * \brief An iterator over the NodeRangeBase container.
 * 
 * The Node%s are iterated over in depth-first order.
 * 
 * This iterator meets the requirements of `InputIterator`. It would also
 * the requirements of `BidirectionalIterator` except that the reference type
 * used by this iterator is not `Node&`. A proxy reference type
 * NodeReferenceProxyBase is used instead.
 * 
 * \tparam Const whether this is a `const` variant of the iterator
 * \tparam Reverse whether this is a reverse or forward iterator
 */
template<typename L, typename N, std::size_t Dim>
template<bool Const, bool Reverse>
class Orthtree<L, N, Dim>::NodeIteratorBase final {
	
private:
	
	friend Orthtree<L, N, Dim>;
	
	template<bool Const_>
	friend class Orthtree<L, N, Dim>::NodeRangeBase;
	
	using ReferenceProxy = NodeReferenceProxyBase<Const>;
	using List = NodeList;
	using ListIterator = std::conditional_t<
			Const,
			typename NodeList::const_iterator,
			typename NodeList::iterator>;
	using ListReference = std::conditional_t<
			Const,
			typename NodeList::const_reference,
			typename NodeList::reference>;
	using OrthtreePointer = std::conditional_t<
			Const,
			Orthtree<L, N, Dim> const*,
			Orthtree<L, N, Dim>*>;
	
	// This type is used by the arrow (->) operator.
	struct PointerProxy {
		
		ReferenceProxy reference;
		
		PointerProxy(ReferenceProxy reference) :
				reference(reference) {
		}
		
		ReferenceProxy* operator->() {
			return &reference;
		}
		
	};
	
	OrthtreePointer _orthtree;
	typename List::difference_type _index;
	
	NodeIteratorBase(
			OrthtreePointer orthtree,
			typename List::difference_type index) :
			_orthtree(orthtree),
			_index(index) {
	}
	
	// Converts this iterator to an iterator over the internal node type.
	ListIterator internalIt() const {
		return _orthtree->_nodes.begin() + _index;
	}
	
public:
	
	// Iterator typedefs.
	using value_type = Node;
	using reference = ReferenceProxy;
	using pointer = PointerProxy;
	using size_type = typename List::size_type;
	using difference_type = typename List::difference_type;
	using iterator_category = std::input_iterator_tag;
	
	NodeIteratorBase() :
			_orthtree(NULL),
			_index(0) {
	}
	
	operator NodeIteratorBase<true, Reverse>() const {
		return NodeIteratorBase<true, Reverse>(_orthtree, _index);
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
		return NodeIteratorBase<Const, !Reverse>(_orthtree, _index + shift);
	}
	
	// Iterator element access methods.
	reference operator*() const {
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
		return lhs._orthtree == rhs._orthtree && lhs._index == rhs._index;
	}
	friend bool operator!=(
			NodeIteratorBase<Const, Reverse> const& lhs,
			NodeIteratorBase<Const, Reverse> const& rhs) {
		return !(lhs == rhs);
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
template<typename L, typename N, std::size_t Dim>
template<bool Const>
class Orthtree<L, N, Dim>::NodeRangeBase final {
	
private:
	
	friend Orthtree<L, N, Dim>;
	
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
			const_iterator,
			iterator>(
			_orthtree,
			_lowerIndex);
	}
	const_iterator cbegin() const {
		return const_iterator(_orthtree, _lowerIndex);
	}
	
	std::conditional_t<Const, const_iterator, iterator> end() const {
		return std::conditional_t<
			Const,
			const_iterator,
			iterator>(
			_orthtree,
			_upperIndex);
	}
	const_iterator cend() const {
		return const_iterator(_orthtree, _upperIndex);
	}
	
	std::conditional_t<Const, const_reverse_iterator, reverse_iterator>
			rbegin() const {
		return std::conditional_t<
			Const,
			const_reverse_iterator,
			reverse_iterator>(
			_orthtree, _upperIndex - 1);
	}
	const_reverse_iterator crbegin() const {
		return const_reverse_iterator(_orthtree, _upperIndex - 1);
	}
	
	std::conditional_t<Const, const_reverse_iterator, reverse_iterator>
			rend() const {
		return std::conditional_t<
			Const,
			const_reverse_iterator,
			reverse_iterator>(
			_orthtree, _lowerIndex - 1);
	}
	const_reverse_iterator crend() const {
		return const_reverse_iterator(_orthtree, _lowerIndex - 1);
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

#include "orthtree.tpp"

#endif

