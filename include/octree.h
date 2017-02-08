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
 * the nodes of the octree (for example, the center of mass of all of the data
 * points contained within a node).
 * 
 * The nested Leaf class is used to contain the discrete data points contained
 * within the Octree. The nested Node class contains the data associated with
 * the nodes themselves. In many ways, the Octree class can be thought of as a
 * container of both Leaf%s and Node%s, although only Leaf%s can be directly
 * added to the Octree.
 * 
 * When an Octree is created, it's possible to specify the maximum number of
 * Leaf%s that can be contained by a Node, as well as the maximum depth that the
 * Octree can reach. In addition, it's possible to choose whether the Octree
 * will automatically readjust its Node%s when one of the Node%s has too many
 * children, or whether the adjustments must be performed manually.
 * 
 * \tparam L the type of data stored at the leaves of the Octree
 * \tparam N the type of data stored at the nodes of the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename L, typename N, std::size_t Dim>
class Octree final {
	
private:
	
	template<bool Const, bool Reverse>
	class LeafIteratorBase;	
	
	template<bool Const>
	class LeafRangeBase;
	
	template<bool Const, bool Reverse>
	class NodeIteratorBase;	
	
	template<bool Const>
	class NodeRangeBase;
	
public:
	
	using LeafRange = LeafRangeBase<false>;
	using ConstLeafRange = LeafRangeBase<true>;
	
	///@{
	/**
	 * \brief Depth-first iterator over all of the Leaf%s contained in the
	 * Octree.
	 */
	using LeafIterator = LeafIterator<false, false>;
	using ConstLeafIterator = LeafIterator<true, false>;
	using ReverseLeafIterator = LeafIterator<false, true>;
	using ConstReverseLeafIterator = LeafIterator<true, true>;
	///@}
	
	using NodeRange = NodeRangeBase<false>;
	using ConstNodeRange = NodeRangeBase<true>;
	
	///@{
	/**
	 * \brief Depth-first iterator over all of the Node%s contained in the
	 * Octree.
	 */
	using NodeIterator = NodeIteratorBase<false, false>;
	using ConstNodeIterator = NodeIteratorBase<true, false>;
	using ReverseNodeIterator = NodeIteratorBase<false, true>;
	using ConstReverseNodeIterator = NodeIteratorBase<true, true>;
	///@}
	
	using LeafList = std::vector<Leaf>;
	using NodeList = std::vector<Node>;
	
private:
	
	struct Leaf final {
		
		L data;
		Vector<Dim> position;
		
		Node(L data, Vector<Dim> position) :
				data(data),
				position(position) {
		}
		
	};
	
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
	
	// A list storing all of the leaves of the octree (it's spelled wrong for
	// consistancy).
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
	 * \param nodeCapacity { the number of Leaf%s that can be stored in
	 * one Node }
	 * \param maxDepth the maximum number of generations of Node%s
	 * \param adjust { whether the Octree should automatically create and
	 * destroy Node%s }
	 */
	Octree(
			Vector<Dim> position,
			Vector<Dim> dimensions,
			std::size_t nodeCapacity = 1,
			std::size_t maxDepth = sizeof(Scalar) * CHAR_BIT,
			bool adjust = true);
	
	LeafRange leafs();
	ConstLeafRange leafs() const;
	ConstLeafRange cleafs() const;
	
	NodeRange nodes();
	ConstNodeRange nodes() const;
	ConstNodeRange cnodes() const;
	
	///@{
	/**
	 * \brief Creates and destroys Node%s so that the Octree has the minimum
	 * necessary Node%s to store all of the Leaf%s.
	 * 
	 * This method takes all Node%s that have more Leaf%s than the capacity and
	 * splits them into child Node%s to store the Leaf%s. It also merges
	 * unnecessary Node%s with their siblings.
	 * 
	 * NodeIterator%s and LeafIterator%s may be invalidated.
	 * 
	 * \param node the Node which will be adjusted (including children)
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
	 * \brief Adds a new Leaf to the Octree.
	 * 
	 * The Octree will search for the appropriate node to which to add the Leaf.
	 * The search starts at the optional Node parameter (if no parameter is
	 * provided, then the search starts at the root). If the Leaf could not be
	 * added (for instance, the position is out of range of the Octree), then
	 * the past-the-end LeafIterator will be returned.
	 * 
	 * NodeIterator%s and LeafIterator%s may be invalidated.
	 * 
	 * \param start a starting guess as to where the Leaf should be placed
	 * \param data the actual data that will be stored at the Leaf
	 * \param position the position of the Leaf
	 * 
	 * \return a tuple containing the NodeIterator to the Node that the
	 * Leaf was added to, and a LeafIterator to the new Leaf
	 */
	std::tuple<NodeIterator, LeafIterator> insert(
			ConstNodeIterator start,
			L const& data,
			Vector<Dim> const& position);
	
	std::tuple<NodeIterator, LeafIterator> insert(
			L const& data,
			Vector<Dim> const& position);
	///@}
	
	///@{
	/**
	 * \brief Removes an Leaf from the Octree.
	 * 
	 * The Octree will search for the appropriate node from which to remove the
	 * Leaf. The search starts at the optional Node parameter (if no parameter
	 * is provided, then the search starts at the root).
	 * 
	 * NodeIterator%s and LeafIterator%s may be invalidated.
	 * 
	 * \param start a starting guess as to where the Leaf should be removed from
	 * \param leaf a LeafIterator to the Leaf that should be removed
	 * 
	 * \return a tuple containing the NodeIterator that the Leaf was removed
	 * from, and the LeafIterator following the removed Leaf
	 */
	std::tuple<NodeIterator, LeafIterator> erase(
			ConstNodeIterator start,
			LeafIterator leaf);
	
	std::tuple<NodeIterator, LeafIterator> erase(
			LeafIterator leaf);
	///@}
	
	///@{
	/**
	 * \brief Changes the position of a Leaf within the Octree.
	 * 
	 * The Octree will search for the appropriate nodes to move the Leaf
	 * between. The search starts at the optional Node parameter (if no
	 * parameter is provided, then the search starts at the root).
	 * 
	 * Node Iterator%s and LeafIterator%s may be invalidated.
	 * 
	 * \param start a starting guess as to where the Leaf should be moved
	 * \param leaf a LeafIterator to the Leaf that should be moved
	 * \param position the new position that the LeafIterator should be moved to
	 * 
	 * \return a tuple containing the NodeIterator that the Leaf was removed
	 * from, the NodeIterator that it was moved to, and the LeafIterator itself
	 */
	std::tuple<NodeIterator, NodeIterator, LeafIterator> move(
			ConstNodeIterator start,
			LeafIterator leaf,
			Vector<Dim> const& position);
	
	std::tuple<NodeIterator, NodeIterator, LeafIterator> move(
			LeafIterator leaf,
			Vector<Dim> const& position);
	///@}
	
	///@{
	/**
	 * \brief Searches for the Node that contains a certain position.
	 * 
	 * This method searches for the unique Node that has no children and still
	 * contains a given position. The search starts at the optional Node
	 * parameter (if no parameter is provided, then the search starts at the
	 * root).
	 * 
	 * \param start an initial guess as to what Node contains the position
	 * \param position the position to search for
	 * 
	 * \return the Node that contains the position
	 */
	NodeIterator find(
			ConstNodeIterator start,
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
	 * \brief Searchs for the Node that contains a certain Leaf.
	 * 
	 * This method searches for the unique Node that has no children and still
	 * contains a given Leaf. The search starts at the optional Node parameter
	 * (if no parameter is provided, then the search starts at the root).
	 * 
	 * \param start an initial guess as to what Node contains the position
	 * \param leaf the Leaf to search for
	 * 
	 * \return the Node that contains the Leaf
	 */
	NodeIterator find(
			ConstNodeIterator start,
			ConstLeafIterator leaf);
	
	NodeIterator find(
			ConstLeafIterator leaf) {
		return find(nodes().begin(), leaf);
	}
	
	ConstNodeIterator find(
			ConstNodeIterator start,
			ConstLeafIterator leaf) const {
		return const_cast<Octree<L, N, Dim>*>(this)->find(start, leaf);
	}
	
	ConstNodeIterator find(
			ConstLeafIterator leaf) const {
		return find(nodes().begin(), leaf);
	}
	///@}
	
};



template<typename L, typename N, std::size_t Dim, bool Const, bool Reverse>
class Octree<L, N, Dim>::LeafIteratorBase<Const, Reverse> final {
	
private:
	
	friend Octree<L, N, Dim>;
	friend Octree<L, N, Dim>::LeafRangeBase<Const>;
	
	using Range = Octree<L, N, Dim>::LeafRange<Const>;
	using List = Octree<L, N, Dim>::LeafList;
	using ListIterator = List::iterator;
	using ListReference = List::reference;
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
	
	ListReference listRef() const {
		return _octree->_leafs[_index];
	}
	ListIterator listIt() const {
		return _octree->_leafs.begin() + _index;
	}
	
public:
	
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
	LeafIteratorBase<Const, !Reverse> reverse() const {
		difference_type shift = !Reverse ? -1 : +1;
		return LeafIteratorBase<Const, !Reverse>(_octree, _index + shift);
	}
	
	Vector<Dim> const& position() const {
		return listRef().position;
	}
	
	reference operator*() const {
		return _octree->_leafs[index].data;
	}
	pointer operator->() const {
		return &_octree->_leafs[index].data;
	}
	reference operator[](difference_type n) const {
		return _octree->_leafs[index + n].data;
	}
	
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



template<typename L, typename N, std::size_t Dim, bool Const>
class Octree<L, N, Dim>::LeafRangeBase<Const> final {
	
private:
	
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
	
	size_type size() const {
		return _upperIndex - _lowerIndex;
	}
	size_type max_size() const {
		return size();
	}
	bool empty() const {
		return _upperIndex == _lowerIndex;
	}
	
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



template<
		typename L, typename N, std::size_t Dim,
		bool Const,
		bool Reverse>
class Octree<L, N, Dim>::NodeIteratorBase<Const, Reverse> {
	
private:
	
	friend Octree<L, N, Dim>;
	friend Octree<L, N, Dim>::NodeRange<Const>;
	
	using Range = Octree<L, N, Dim>::NodeRange<Const>;
	using List = Octree<L, N, Dim>::NodeList;
	using ListIterator = List::iterator;
	using ListReference = List::reference;
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
	
	ListReference listRef() const {
		return _octree->_nodes[_index];
	}
	ListIterator listIt() const {
		return _octree->_nodes.begin() + _index;
	}
	
public:
	
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
	NodeIteratorBase<Const, !Reverse> reverse() const {
		difference_type shift = !Reverse ? -1 : +1;
		return NodeIteratorBase<Const, !Reverse>(
			_octree,
			_index + shift);
	}
	
	Vector<Dim> const& position() const {
		return listRef().position;
	}
	Vector<Dim> const& dimensions() const {
		return listRef().dimensions;
	}
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
	bool contains(Octree<L, N, Dim>::ConstLeafIterator leaf) const {
		return
			leaf._index >= listRef().leafIndex &&
			leaf._index < listRef().leafIndex + listRef().leafCount;
	}
	bool canHoldLeafs(
			Octree<L, N, Dim>::LeafRange::difference_type n = 0) const {
		return
			listRef().dataCount + n < _octree->_nodeCapacity ||
			listRef().depth >= _octree->_maxDepth;
	}
	
	bool hasParent() const {
		return listRef().hasParent;
	}
	NodeIteratorBase<Const, Reverse> parent() const {
		return NodeIteratorBase<Const, Reverse>(
			_octree,
			_index + listRef().parentIndex);
	}
	
	bool hasChildren() const {
		return listRef().hasChildren;
	}
	NodeIteratorBase<Const, Reverse> child(
			size_type childIndex) const {
		return NodeIteratorBase<Const, Reverse>(
			_octree,
			_index + listRef().childIndices[childIndex]);
	}
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
	NodeIteratorBase<Const, Reverse> child(
			ConstLeafIterator leaf) const {
		for (size_type childIndex = 0; childIndex < (1 << Dim); ++childIndex) {
			NodeIteratorBase<Const, Reverse> child = child(childIndex);
			if (child.contains(leaf)) {
				return child;
			}
		}
		return
			Reverse ?
			_octree->nodes().rend() :
			_octree->nodes().end();
	}
	
	NodeRangeBase<Const> children() const;
	NodeRangeBase<Const> nodes() const;
	LeafRangeBase<Const> leafs() const;
	
	reference operator*() const {
		return _octree->_nodes[_index].data;
	}
	pointer operator->() const {
		return &_octree->_nodes[_index].data;
	}
	reference operator[](difference_type n) const {
		return _octree->_nodes[_index + n].data;
	}
	
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



template<
		typename L, typename N, std::size_t Dim,
		bool Const>
class Octree<L, N, Dim>::NodeRangeBase<Const> {
	
private:
	
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
	
	size_type size() const {
		return _upperIndex - _lowerIndex;
	}
	size_type max_size() const {
		return size();
	}
	bool empty() const {
		return _upperIndex == _lowerIndex;
	}
	
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

