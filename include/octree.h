#ifndef __NBODY_OCTREE_H_
#define __NBODY_OCTREE_H_

#include <climits>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <vector>

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
	
private:
	
	// These private classes are used to construct const-variants and
	// reverse-variants of the public range and iterator classes.
	
	template<bool Const, bool Reverse>
	class LeafIteratorBase;	
	
	template<bool Const>
	class LeafRangeBase;
	
	template<bool Const, bool Reverse>
	class NodeIteratorBase;	
	
	template<bool Const>
	class NodeRangeBase;
	
public:
	
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
	using LeafIterator = LeafIterator<false, false>;
	using ConstLeafIterator = LeafIterator<true, false>;
	using ReverseLeafIterator = LeafIterator<false, true>;
	using ConstReverseLeafIterator = LeafIterator<true, true>;
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
	
	using LeafList = std::vector<Leaf>;
	using NodeList = std::vector<Node>;
	
	// This class packages leaf data with a position.
	struct Leaf final {
		
		L data;
		Vector<Dim> position;
		
		Node(L data, Vector<Dim> position) :
				data(data),
				position(position) {
		}
		
	};
	
	// This class packages node data within the node heirarchy.
	struct Node final {
		
		// The depth of this node within the octree (0 for root, and so on).
		std::size_t depth;
		
		// Whether this node has any children.
		bool hasChildren;
		// The indices of the children of this node, stored relative to the
		// index of this node. The last entry points to the next sibling of this
		// node, and is used to determine the total size of all of this node's
		// children.
		std::size_t childIndices[(1 << Dim) + 1];
		
		// Whether this node has a parent.
		bool hasParent;
		// The relative index of the parent of this node.
		std::ptrdiff_t parentIndex;
		// Which child # of its parent this node is. (0th child, 1st child,
		// etc).
		std::size_t siblingIndex;
		
		// The number of leaves that this node contains. This includes leaves
		// stored by all descendants of this node.
		std::size_t leafCount;
		// The index within the octree's leaf array that this node's leaves are
		// located at.
		std::size_t leafIndex;
		
		// The section of space that this node encompasses.
		Vector<Dim> position;
		Vector<Dim> dimensions;
		
		// The data stored at the node itself.
		N data;
		
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
				data() {
		}
		
	};
	
	// A list storing all of the leafs of the octree.
	LeafList _leafs;
	
	// A list storing all of the nodes of the octree.
	NodeList _nodes;
	
	// The number of leaves to store at a single node of the octree.
	std::size_t const _nodeCapacity;
	
	// The maximum depth of the octree.
	std::size_t const _maxDepth;
	
	// Whether the tree should be automatically readjust itself so that each
	// node has less leaves than the node capacity, as well as having as few
	// children as possible. If this is false, then the adjust() method has to
	// be called to force an adjustment.
	bool const _adjust;
	
	
	
	// Divides a node into a set of subnodes and partitions its leaves between
	// them. This function may reorganize the leaf vector (some leaf iterators
	// may become invalid).
	NodeIterator createChildren(ConstNodeIterator node);
	
	// Destroys all descendants of a node and takes their leaves into the node.
	// This function will not reorganize the leaf vector (all leaf iterators
	// will remain valid).
	NodeIterator destroyChildren(ConstNodeIterator node);
	
	// Adds a leaf to a specific node.
	LeafIterator insertAt(ConstNodeIterator node, Leaf const& leaf);
	
	// Removes a leaf from a node.
	LeafIterator eraseAt(ConstNodeIterator node, ConstLeafIterator leaf);
	
	// Moves a leaf from this node to another one.
	LeafIterator moveAt(
			ConstNodeIterator sourceNode,
			ConstNodeIterator destNode,
			ConstLeafIterator sourceLeaf);
	
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
			std::size_t nodeCapacity = 1,
			std::size_t maxDepth = sizeof(Scalar) * CHAR_BIT,
			bool adjust = true);
	
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
		return const_cast<Octree<L, N, Dim>*>(this)->find(start, leaf);
	}
	
	ConstNodeIterator find(
			ConstLeafIterator leaf) const {
		return find(nodes().begin(), leaf);
	}
	///@}
	
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
template<typename L, typename N, std::size_t Dim, bool Const, bool Reverse>
class Octree<L, N, Dim>::LeafIteratorBase<Const, Reverse> final {
	
private:
	
	friend Octree<L, N, Dim>;
	template<bool Const>
	friend Octree<L, N, Dim>::LeafRangeBase<Const>;
	template<bool Const>
	friend Octree<L, N, Dim>::NodeIteratorBase<Const>;
	
	// These typedefs reference the underlying lists in the Octree class that
	// actually store the leaves.
	using Range = Octree<L, N, Dim>::LeafRange<Const>;
	using List = Octree<L, N, Dim>::LeafList;
	using ListIterator = std::conditional_t<
			Const,
			List::const_iterator,
			List::iterator>;
	using ListReference = std::conditional_t<
			Const,
			List::const_reference,
			List::reference>;
	
	using OctreePointer = std::conditional_t<
			Const,
			Octree<L, N, Dim> const*,
			Octree<L, N, Dim>*>;
	
	OctreePointer const _octree;
	Range::difference_type _index;
	
	LeafIteratorBase(OctreePointer octree, Range::difference_type index) :
			_octree(octree),
			_index(index) {
	}
	
	// These methods provide convenient access to the underlying list.
	ListReference listRef() const {
		return _octree->_leafs[_index];
	}
	ListIterator listIt() const {
		return _octree->_leafs.begin() + _index;
	}
	
public:
	
	// Iterator typedefs.
	using value_type = Range::value_type;
	using reference = std::conditional_t<
			Const,
			Range::const_reference,
			Range::reference>;
	using pointer = std::conditional_t<
			Const,
			Range::const_pointer,
			Range::pointer>;
	using size_type = Range::size_type;
	using difference_type = Range::difference_type;
	using iterator_category = std::random_access_iterator_tag;
	
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
	
	/**
	 * \brief Gets the position of the data pointed to by this iterator.
	 */
	Vector<Dim> const& position() const {
		return listRef().position;
	}
	
	// Iterator element access methods.
	reference operator*() const {
		return _octree->_leafs[index].data;
	}
	pointer operator->() const {
		return &_octree->_leafs[index].data;
	}
	reference operator[](difference_type n) const {
		return _octree->_leafs[index + n].data;
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
	
	LeafIteratorBase<Const, Reverse>& operator+=(difference_type n) {
		difference_type shift = Reverse ? -n : +n;
		_index += shift;
		return *this;
	}
	LeafIteratorBase<Const, Reverse>& operator-=(difference_type n) {
		difference_type shift = Reverse ? +n : -n;
		_index += shift;
		return *this;
	}
	
	// Iterator arithmetic methods.
	friend LeafIteratorBase<Const, Reverse> operator+(
			LeafIteratorBase<Const, Reverse> it,
			difference_type n) {
		LeafIteratorBase<Const, Reverse> result = *this;
		result += n;
		return result;
	}
	friend LeafIteratorBase<Const, Reverse> operator+(
			difference_type n,
			LeafIteratorBase<Const, Reverse> it) {
		return it + n;
	}
	friend LeafIteratorBase<Const, Reverse> operator-(
			LeafIteratorBase<Const, Reverse> it,
			difference_type n) {
		LeafIteratorBase<Const, Reverse> result = *this;
		result -= n;
		return result;
	}
	
	friend difference_type operator-(
			LeafIteratorBase<Const, Reverse> const& lhs,
			LeafIteratorBase<Const, Reverse> const& rhs) {
		return Reverse ? (rhs._index - lhs._index) : (lhs._index - rhs._index);
	}
	
	// Iterator comparison methods.
	friend bool operator==(
			LeafIteratorBase<Const, Reverse> const& lhs,
			LeafIteratorBase<Const, Reverse> const& rhs) {
		return lhs._index == rhs._index;
	}
	friend bool operator!=(
			LeafIteratorBase<Const, Reverse> const& lhs,
			LeafIteratorBase<Const, Reverse> const& rhs) {
		return !(lhs == rhs);
	}
	friend bool operator<(
			LeafIteratorBase<Const, Reverse> const& lhs,
			LeafIteratorBase<Const, Reverse> const& rhs) {
		return Reverse ? (rhs._index < lhs._index) : (lhs._index < rhs._index);
	}
	friend bool operator>(
			LeafIteratorBase<Const, Reverse> const& lhs,
			LeafIteratorBase<Const, Reverse> const& rhs) {
		return rhs < lhs;
	}
	friend bool operator<=(
			LeafIteratorBase<Const, Reverse> const& lhs,
			LeafIteratorBase<Const, Reverse> const& rhs) {
		return !(lhs > rhs);
	}
	friend bool operator>=(
			LeafIteratorBase<Const, Reverse> const& lhs,
			LeafIteratorBase<Const, Reverse> const& rhs) {
		return !(lhs < rhs);
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
template<typename L, typename N, std::size_t Dim, bool Const>
class Octree<L, N, Dim>::LeafRangeBase<Const> final {
	
private:
	
	friend Octree<L, N, Dim>;
	template<bool Const, bool Reverse>
	friend Octree<L, N, Dim>::NodeIteratorBase<Const, Reverse>;
	
	using OctreePointer = std::conditional_t<
			Const,
			Octree<L, N, Dim> const*,
			Octree<L, N, Dim>*>;
	
	OctreePointer const _octree;
	std::size_t _lowerIndex;
	std::size_t _upperIndex;
	
	LeafRangeBase(
			OctreePointer octree,
			std::size_t lowerIndex,
			std::size_t upperIndex) :
			_octree(octree),
			_lowerIndex(lowerIndex),
			_upperIndex(upperIndex) {
	}
	
public:
	
	// Container typedefs.
	using value_type = L;
	using reference = L&;
	using const_reference = L const&;
	using pointer = L*;
	using pointer = L const*;
	using iterator = Octree<L, N, Dim>::LeafIterator;
	using const_iterator = Octree<L, N, Dim>::ConstLeafIterator;
	using reverse_iterator = Octree<L, N, Dim>::ReverseLeafIterator;
	using const_reverse_iterator = Octree<L, N, Dim>::ConstReverseLeafIterator;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	
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
template<
		typename L, typename N, std::size_t Dim,
		bool Const,
		bool Reverse>
class Octree<L, N, Dim>::NodeIteratorBase<Const, Reverse> final {
	
private:
	
	friend Octree<L, N, Dim>;
	template<bool Const>
	friend Octree<L, N, Dim>::NodeRange<Const>;
	
	// These typedefs reference the underlying lists in the Octree class that
	// actually store the nodes.
	using Range = Octree<L, N, Dim>::NodeRange<Const>;
	using List = Octree<L, N, Dim>::NodeList;
	using ListIterator = std::conditional_t<
			Const,
			List::const_iterator,
			List::iterator>;
	using ListReference = std::conditional_t<
			Const,
			List::const_reference,
			List::reference>;
	
	using OctreePointer = std::conditional_t<
			Const,
			Octree<L, N, Dim> const*,
			Octree<L, N, Dim>*>;
	
	OctreePointer const _octree;
	Range::difference_type _index;
	
	NodeIteratorBase(OctreePointer octree, Range::difference_type index) :
			_octree(octree),
			_index(index) {
	}
	
	// These methods provide convenient access to the underlying list.
	ListReference listRef() const {
		return _octree->_nodes[_index];
	}
	ListIterator listIt() const {
		return _octree->_nodes.begin() + _index;
	}
	
public:
	
	// Iterator typedefs.
	using value_type = Range::value_type;
	using reference = std::conditional_t<
			Const,
			Range::const_reference,
			Range::reference>;
	using pointer = std::conditional_t<
			Const,
			Range::const_pointer,
			Range::pointer>;
	using size_type = Range::size_type;
	using difference_type = Range::difference_type;
	using iterator_category = std::random_access_iterator_tag;
	
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
		return NodeIteratorBase<Const, !Reverse>(
			_octree,
			_index + shift);
	}
	
	/**
	 * \brief Gets the "upper-left" corner of the node pointed to by this
	 * iterator.
	 */
	Vector<Dim> const& position() const {
		return listRef().position;
	}
	/**
	 * \brief Gets the size of the node pointed to by this iterator.
	 */
	Vector<Dim> const& dimensions() const {
		return listRef().dimensions;
	}
	/**
	 * \brief Returns whether a certain point is contained within the node
	 * pointed to by this iterator.
	 */
	bool contains(Vector<Dim> const& point) const {
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (!(
					point[dim] >= position()[dim] &&
					point[dim] < position()[dim] + dimensions()[dim])) {
				return false;
			}
		}
		return true;
	}
	/**
	 * \brief Returns whether a certain leaf is contained within the node
	 * pointed to by this iterator.
	 */
	bool contains(Octree<L, N, Dim>::ConstLeafIterator leaf) const {
		return
			leaf._index >= listRef().leafIndex &&
			leaf._index < listRef().leafIndex + listRef().leafCount;
	}
	/**
	 * \brief Returns whether the node is able to hold a certain number of
	 * additional leaves without being over capacity.
	 */
	bool canHoldLeafs(
			Octree<L, N, Dim>::LeafRange::difference_type n = 0) const {
		return
			listRef().dataCount + n < _octree->_nodeCapacity ||
			listRef().depth >= _octree->_maxDepth;
	}
	
	/**
	 * \brief Returns whether this node has a parent.
	 * 
	 * If this method returns false, then the node must be the root of the
	 * Octree.
	 */
	bool hasParent() const {
		return listRef().hasParent;
	}
	/**
	 * \brief Gets an iterator pointing to the parent of this node.
	 * 
	 * If this node has no parent (that is, this node is the root node), then
	 * the result is undefined.
	 */
	NodeIteratorBase<Const, Reverse> parent() const {
		return NodeIteratorBase<Const, Reverse>(
			_octree,
			_index + listRef().parentIndex);
	}
	
	/**
	 * \brief Returns whether this node has any children.
	 */
	bool hasChildren() const {
		return listRef().hasChildren;
	}
	/**
	 * \brief Returns one of the children of this node by index.
	 * 
	 * By default, a node has either `2 ** Dim` or `0` children. If this node
	 * has no children, then the result is undefined.
	 * 
	 * \param childIndex the index of the child to return
	 */
	NodeIteratorBase<Const, Reverse> child(
			size_type childIndex) const {
		return NodeIteratorBase<Const, Reverse>(
			_octree,
			_index + listRef().childIndices[childIndex]);
	}
	/**
	 * \brief Returns one of the children of this node by position.
	 * 
	 * This method returns the child of this node that contains the given point.
	 * If the point is outside of the bounds of the current node, then it is
	 * automatically normalized to be within the bounds.
	 * 
	 * \param point the position contained by the child to return
	 */
	NodeIteratorBase<Const, Reverse> child(
			Vector<Dim> point) const {
		size_type childIndex = 0;
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (point[dim] >= position()[dim] + dimensions()[dim] / 2.0) {
				childIndex += (1 << dim);
			}
		}
		return child(childIndex);
	}
	/**
	 * \brief Returns one of the children of this node by the leaf that it
	 * contains.
	 * 
	 * This method returns the child of this node that contains the given leaf.
	 * If the leaf is not contained within the current node, then the result is
	 * the past-the-end iterator of the Octree::nodes() container.
	 * 
	 * \param leaf an iterator to the leaf contained by the child to return
	 */
	NodeIteratorBase<Const, Reverse> child(
			ConstLeafIterator leaf) const {
		for (size_type childIndex = 0; childIndex < (1 << Dim); ++childIndex) {
			NodeIteratorBase<Const, Reverse> child = child(childIndex);
			if (child.contains(leaf)) {
				return child;
			}
		}
		return _octree->nodes().end();
	}
	
	/**
	 * \brief Returns a list of all of the descendants of this node, but not
	 * including this node itself.
	 */
	NodeRangeBase<Const> children() const;
	/**
	 * \brief Returns a list containing this node as well as all of its
	 * descendants.
	 */
	NodeRangeBase<Const> nodes() const;
	/**
	 * \brief Returns a list of all of the leaves contained within this node.
	 */
	LeafRangeBase<Const> leafs() const;
	
	// Iterator element access methods.
	reference operator*() const {
		return _octree->_nodes[_index].data;
	}
	pointer operator->() const {
		return &_octree->_nodes[_index].data;
	}
	reference operator[](difference_type n) const {
		return _octree->_nodes[_index + n].data;
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
	
	NodeIteratorBase<Const, Reverse>& operator+=(difference_type n) {
		difference_type shift = Reverse ? -n : +n;
		_index += shift;
		return *this;
	}
	NodeIteratorBase<Const, Reverse>& operator-=(difference_type n) {
		difference_type shift = Reverse ? +n : -n;
		_index += shift;
		return *this;
	}
	
	// Iterator arithmetic methods.
	friend NodeIteratorBase<Const, Reverse> operator+(
			NodeIteratorBase<Const, Reverse> it,
			difference_type n) {
		NodeIteratorBase<Const, Reverse> result = *this;
		result += n;
		return result;
	}
	friend NodeIteratorBase<Const, Reverse> operator+(
			difference_type n,
			NodeIteratorBase<Const, Reverse> it) {
		return it + n;
	}
	friend NodeIteratorBase<Const, Reverse> operator-(
			NodeIteratorBase<Const, Reverse> it,
			difference_type n) {
		NodeIteratorBase<Const, Reverse> result = *this;
		result -= n;
		return result;
	}
	
	friend difference_type operator-(
			NodeIteratorBase<Const, Reverse> const& lhs,
			NodeIteratorBase<Const, Reverse> const& rhs) {
		return Reverse ? rhs._index - lhs._index : lhs._index - rhs._index;
	}
	
	// Iterator comparison methods.
	friend bool operator==(
			NodeIteratorBase<Const, Reverse> const& lhs,
			NodeIteratorBase<Const, Reverse> const& rhs) {
		return lhs._index == rhs._index;
	}
	friend bool operator!=(
			NodeIteratorBase<Const, Reverse> const& lhs,
			NodeIteratorBase<Const, Reverse> const& rhs) {
		return !(lhs == rhs);
	}
	friend bool operator<(
			NodeIteratorBase<Const, Reverse> const& lhs,
			NodeIteratorBase<Const, Reverse> const& rhs) {
		return Reverse ? rhs._index < lhs._index : lhs._index < rhs._index;
	}
	friend bool operator>(
			NodeIteratorBase<Const, Reverse> const& lhs,
			NodeIteratorBase<Const, Reverse> const& rhs) {
		return rhs < lhs;
	}
	friend bool operator<=(
			NodeIteratorBase<Const, Reverse> const& lhs,
			NodeIteratorBase<Const, Reverse> const& rhs) {
		return !(lhs > rhs);
	}
	friend bool operator>=(
			NodeIteratorBase<Const, Reverse> const& lhs,
			NodeIteratorBase<Const, Reverse> const& rhs) {
		return !(lhs < rhs);
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
template<
		typename L, typename N, std::size_t Dim,
		bool Const>
class Octree<L, N, Dim>::NodeRangeBase<Const> final {
	
private:
	
	friend Octree<L, N, Dim>;
	
	using OctreePointer = std::conditional_t<
			Const,
			Octree<L, N, Dim> const*,
			Octree<L, N, Dim>*>;
	
	OctreePointer const _octree;
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
	using pointer = N const*;
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

