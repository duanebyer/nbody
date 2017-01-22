#ifndef __NBODY_OCTREE_H_
#define __NBODY_OCTREE_H_

#include <algorithm>
#include <climits>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "tensor.h"

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
	
public:
	
	// Type declarations have to be forward declared at the top. The rest of
	// interface is declared in a second public section below.
	
	class Node;
	class Leaf;
	
	///@{
	/**
	 * \brief Depth-first iterator over all of the Node%s contained in the
	 * Octree.
	 */
	using NodeIterator = typename std::vector<Node>::iterator;
	using ConstNodeIterator = typename std::vector<Node>::const_iterator;
	///@}
	///@{
	/**
	 * \brief Depth-first iterator over all of the Leaf%s contained in the
	 * Octree.
	 */
	using LeafIterator = typename std::vector<Leaf>::iterator;
	using ConstLeafIterator = typename std::vector<Leaf>::const_iterator;
	///@}
	
private:
	
	// A list storing all of the nodes of the octree.
	std::vector<Node> _nodes;
	
	// A list storing all of the leaves of the octree (it's spelled wrong for
	// consistancy).
	std::vector<Leaf> _leafs;
	
	// The number of leaves to store at a single node of the octree.
	std::size_t const _nodeCapacity;
	
	// The maximum depth of the octree.
	std::size_t const _maxDepth;
	
	// Whether the tree should be automatically readjust itself so that each
	// node has less leaves than the node capacity, as well as having as few
	// children as possible. If this is false, then the adjust() method has to
	// be called to force an adjustment.
	bool const _adjust;
	
	
	
	// Returns whether the node can accomodate a change in the number of leaves
	// points by 'n' and still hold all of the leaves without needing to
	// subdivide.
	bool canHoldLeafs(ConstNodeIterator node, std::ptrdiff_t n = 0) const {
		// If the depth of the node is larger than the max, then it has infinite
		// capacity.
		return
			node->_dataCount + n < _nodeCapacity ||
			node->_depth >= _maxDepth;
	}
	
	// Divides a node into a set of subnodes and partitions its leaves between
	// them. This function may reorganize the leaf vector (some leaf iterators
	// may become invalid).
	NodeIterator createChildren(ConstNodeIterator node) {
		// Create the 2^Dim child nodes inside the parent octree.
		auto firstChild = _nodes.insert(node, 1 << Dim, Node());
		
		// Update the node iterator as it has been invalidated.
		node = firstChild - 1;
		
		// Loop through the new children, and set up their various properties.
		Vector<Dim> childDimensions = node->_dimensions / 2;
		Vector<Dim> midpoint = node->_position + childDimensions;
		auto child = firstChild;
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
			child->_depth = node->_depth + 1;
			child->_hasParent = true;
			child->_parentIndex = -((std::ptrdiff_t) index + 1);
			child->_siblingIndex = index;
			child->_leafIndex = node->_leafIndex + node->_leafCount;
			child->_dimensions = childDimensions;
			child->_position = node->_position;
			// Shift the child position depending on which child it is.
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				if ((1 << dim) & index) {
					child->_position[dim] = midpoint[dim];
				}
			}
			++child;
			
			// Add the child to this node.
			node->_childIndices[index] = index + 1;
		}
		
		// Go through the parent, grandparent, great-grandparent, ...  of this
		// node and update their child indices.
		auto parent = node;
		while (parent->_hasParent) {
			std::size_t siblingIndex = parent->_siblingIndex;
			parent += parent->_parentIndex;
			while (++siblingIndex <= (1 << Dim)) {
				parent->_childIndices[siblingIndex] += (1 << Dim);
				auto child = parent + parent->_childIndices[siblingIndex];
				child->_parentIndex -= (1 << Dim);
			}
		}
		
		// Distribute the leaves of this node to the children.
		for (std::size_t index = 0; index < node->_leafCount; ++index) {
			// Figure out which node the leaf belongs to.
			std::size_t childIndex = 0;
			auto leaf = _leafs.begin() + node->_leafIndex + index;
			Vector<Dim> const& position = leaf->position();
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				if (position[dim] >= midpoint[dim]) {
					childIndex += (1 << dim);
				}
			}
			moveAt(node, firstChild + childIndex, leaf);
		}
		
		return node;
	}
	
	// Destroys all descendants of a node and takes their leaves into the node.
	// This function will not reorganize the leaf vector (all leaf iterators
	// will remain valid).
	NodeIterator destroyChildren(ConstNodeIterator node) {
		// Determine how many children, grandchildren, great-grandchildren, ...
		// of this node.
		std::size_t numDescendants = node->_childIndices[1 << Dim];
		
		// Destroy the subnodes and update iterators.
		auto nextNode = _nodes->erase(node + 1, node + 1 + numDescendants);
		node = nextNode - 1;
		
		// Go through the parent, grandparent, great-grandparent, ... of this
		// node and update their child indices.
		auto parent = node;
		while (parent->_hasParent) {
			std::size_t siblingIndex = parent->_siblingIndex;
			parent += parent->_parentIndex;
			while (++siblingIndex <= (1 << Dim)) {
				parent->_childIndices[siblingIndex] -= numDescendants;
				auto child = parent + parent->_childIndices[siblingIndex];
				child->_parentIndex += numDescendants;
			}
		}
		
		return node;
	}
	
	// Adds a leaf to a specific node.
	LeafIterator insertAt(
			ConstNodeIterator node,
			Leaf const& leaf) {
		// Add the leaf to the master list of leaves in the octree and update
		// internal variables.
		auto newLeaf = _leafs.insert(
			_leafs.begin + node->_leafIndex + node->_leafCount);
		
		// Loop through the rest of the nodes and increment their leaf indices
		// so that they still refer to the correct location in the leaf vector.
		auto currentNode = node;
		while (++currentNode != _nodes.end()) {
			++currentNode->_leafIndex;
		}
		
		// Also loop through all ancestors and increment their leaf counts.
		auto parent = node;
		while (parent->_hasParent) {
			++parent->_leafCount;
			parent += parent->_parentIndex;
		}
		
		return newLeaf;
	}
	
	// Removes a leaf from a node.
	LeafIterator eraseAt(
			ConstNodeIterator node,
			ConstLeafIterator leaf) {
		// Remove the leaf from the master octree leaf vector.
		auto next = _leafs.erase(leaf);
		
		// Loop through the rest of the nodes and increment their leaf indices
		// so that they still refer to the correct location in the leaf vector.
		auto currentNode = node;
		while (++currentNode != _nodes.end()) {
			--currentNode->_leafIndex;
		}
		
		// Loop through all of the ancestors of this node and decremement their
		// leaf counts.
		auto parent = node;
		while (parent->_hasParent) {
			--parent->_leafCount;
			parent += parent->_parentIndex;
		}
		
		return next;
	}
	
	// Moves a leaf from this node to another one.
	LeafIterator moveAt(
			ConstNodeIterator sourceNode,
			ConstNodeIterator destNode,
			ConstLeafIterator sourceLeaf) {
		// Reinsert the leaf into the leaf vector in its new position.
		auto destLeaf =
			_leafs.begin() + destNode->_leafIndex + destNode->_leafCount;
		bool inverted = sourceLeaf > destLeaf;
		auto firstLeaf = inverted ? destLeaf : sourceLeaf;
		auto lastLeaf = inverted ? sourceLeaf : destLeaf;
		
		std::size_t sourceLeafIndex = sourceLeaf - _leafs.begin();
		std::size_t destLeafIndex = destLeaf - _leafs.begin();
		
		std::rotate(firstLeaf, sourceLeaf + !inverted, lastLeaf + inverted);
		
		// Adjust the ancestors of the source node.
		auto sourceParentNode = sourceNode;
		std::size_t sourceChildIndex = 0;
		while (!(
				destLeafIndex >= sourceParentNode->_leafIndex &&
				destLeafIndex < sourceParentNode->_leafIndex +
				                sourceParentNode->_leafCount)) {
			--sourceParentNode->_leafCount;
			std::size_t siblingIndex = sourceParentNode->_siblingIndex;
			sourceParentNode += sourceParentNode->_parentIndex;
			sourceChildIndex += sourceParentNode->_childIndices[siblingIndex];
		}
		
		// Adjust the ancestors of the destination node.
		auto destParentNode = destNode;
		std::size_t destChildIndex = 0;
		while (!(
				sourceLeafIndex >= destParentNode->_leafIndex &&
				sourceLeafIndex < destParentNode->_leafIndex +
				                  destParentNode->_leafCount)) {
			++destParentNode->_leafCount;
			std::size_t siblingIndex = destParentNode->_siblingIndex;
			destParentNode += destParentNode->_parentIndex;
			destChildIndex += destParentNode->_childIndices[siblingIndex];
		}
		
		// Both sourceParentNode and destParentNode should be equal to the same
		// node: the most recent common ancestor of both the source and
		// destination.
		
		// Adjust the nodes in between the source and destination node.
		std::ptrdiff_t invertedSign = inverted ? -1 : +1;
		for (
				std::size_t nodeIndex = sourceChildIndex;
				nodeIndex < destChildIndex;
				nodeIndex += invertedSign) {
			auto node = sourceParentNode + nodeIndex;
			node->_leafIndex -= invertedSign;
		}
		
		return _leafs.begin() + destLeafIndex;
	}
	
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
			bool adjust = true) :
			_nodes(),
			_leafs(),
			_nodeCapacity(1),
			_maxDepth(maxDepth),
			_adjust(adjust) {
		Node root;
		root._position = position;
		root._dimensions = dimensions;
		_nodes.push_back(root);
	}
	
	
	
	///@{
	/**
	 * \brief Gets a NodeIterator to the root Node of the Octree.
	 * 
	 * The root Node is the only Node that has no parent. Its dimensions are the
	 * same as the dimensions of the whole Octree.
	 */
	NodeIterator root() {
		return _nodes.begin();
	}
	ConstNodeIterator root() const {
		return _nodes.begin();
	}
	///@}
	
	///@{
	/**
	 * \brief Standard iterator method for depth-first iteration order over
	 * Node%s.
	 */
	NodeIterator nodeBegin() {
		return _nodes.begin();
	}
	ConstNodeIterator nodeBegin() const {
		return _nodes.begin();
	}
	NodeIterator nodeEnd() {
		return _nodes.end();
	}
	ConstNodeIterator nodeEnd() const {
		return _nodes.end();
	}
	///@}
	
	///@{
	/**
	 * \brief Gets a NodeIterator to the parent of a Node.
	 * 
	 * If the Node has no parent (because it is the root), then the past-the-end
	 * NodeIterator is returned.
	 */
	NodeIterator parent(ConstNodeIterator node) {
		if (!node->_hasParent) {
			return _nodes.end();
		}
		else {
			return node + node->_parentIndex;
		}
	}
	ConstNodeIterator parent(ConstNodeIterator node) const {
		return const_cast<Octree<L, N, Dim>*>(this)->parent(node);
	}
	///@}
	
	///@{
	/**
	 * \brief Gets a NodeIterator to a child of an Node.
	 * 
	 * If the Node has no children, then the past-the-end NodeIterator is
	 * returned.
	 */
	NodeIterator child(ConstNodeIterator node, std::size_t index) {
		if (!node->_hasChildren) {
			return _nodes.end();
		}
		else {
			return node + node->_childIndices[index];
		}
	}
	ConstNodeIterator child(ConstNodeIterator node, std::size_t index) const {
		return const_cast<Octree<L, N, Dim>*>(this)->child(node, index);
	}
	///@}
	
	///@{
	/**
	 * \brief Standard iterator method for depth-first iteration order over
	 * Leaf%s.
	 */
	LeafIterator leafBegin() {
		return _leafs.begin();
	}
	ConstLeafIterator leafBegin() const {
		return _leafs.begin();
	}
	LeafIterator leafEnd() {
		return _leafs.end();
	}
	ConstLeafIterator leafEnd() const {
		return _leafs.end();
	}
	///@}
	
	///@{
	/**
	 * \brief Provides depth-first iteration bounds for all of the Leaf%s
	 * contained within a specific Node (or its children).
	 */
	LeafIterator leafBegin(ConstNodeIterator node) {
		return _leafs.begin() + node->_leafIndex;
	}
	ConstLeafIterator leafBegin(ConstNodeIterator node) const {
		return _leafs.begin() + node->_leafIndex;
	}
	LeafIterator leafEnd(ConstNodeIterator node) {
		return _leafs.begin() + node->_leafIndex + node->_leafCount;
	}
	ConstLeafIterator leafEnd(ConstNodeIterator node) const {
		return _leafs.begin() + node->_leafIndex + node->_leafCount;
	}
	///@}
	
	
	
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
	bool adjust(ConstNodeIterator node) {
		bool result = false;
		// If the node doesn't have children but should, then make them.
		if (!node->_hasChildren && !canHoldLeafs(node, 0)) {
			node = createChildren(node);
			result = true;
		}
		// If the node does have children but shouldn't, then remove them.
		else if (node->_hasChildren && canHoldLeafs(node, 0)) {
			node = destroyChildren(node);
			result = true;
		}
		// Then, adjust all of this node's children as well.
		if (node->_hasChildren) {
			auto begin = _nodes.begin();
			for (std::size_t index = 0; index < (1 << Dim); ++index) {
				auto child = node + node->_childIndices[index];
				bool nextResult = adjust(child);
				result = result || nextResult;
			}
		}
		return result;
	}
	
	bool adjust() {
		return adjust(root());
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
			Leaf const& data,
			Vector<Dim> const& position) {
		// Find the node with the correct position, and insert the leaf into
		// that node.
		Leaf leaf(data, position);
		auto node = find(start, leaf.position());
		if (node == _nodes.end()) {
			return _leafs.end();
		}
		// Create children if the node doesn't have the capacity to store
		// this leaf.
		if (_adjust && !canHoldLeafs(node, +1)) {
			node = createChildren(node);
			node = find(node, leaf.position());
		}
		return std::make_tuple(node, insertAt(node, leaf));
	}
	
	std::tuple<NodeIterator, LeafIterator> insert(
			Leaf const& data,
			Vector<Dim> const& position) {
		return insert(root(), data, position);
	}
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
			LeafIterator leaf) {
		// Find the node that contains this leaf, and then erase the leaf from
		// that node.
		auto node = find(start, leaf);
		if (node == _nodes.end()) {
			return _leafs.end();
		}
		// If the parent of this node doesn't need to be divided into subnodes
		// anymore, then merge its children together.
		while (
				_adjust &&
				node->_hasParent &&
				canHoldLeafs(node + node->_parentIndex, -1)) {
			node = destroyChildren(node + node->_parentIndex);
		}
		return std::make_tuple(node, eraseAt(node, leaf));
	}
	
	std::tuple<NodeIterator, LeafIterator> erase(
			LeafIterator leaf) {
		return erase(root(), leaf);
	}
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
			Vector<Dim> const& position) {
		// Find the source node that contains the leaf, and the target node with
		// the correct position.
		auto source = find(start, leaf);
		auto dest = find(start, position);
		if (source == _nodes.end() || dest == _nodes.end()) {
			return _leafs.end();
		}
		// If the source and the destination are distinct, then check to make
		// sure that they remain within the node capcity.
		if (_adjust && source != dest) {
			while (
					source->_hasParent &&
					canHoldLeafs(source + source->_parentIndex, -1)) {
				std::size_t destIndex = dest - _nodes.begin();
				if (dest > source) {
					destIndex -= (1 << Dim);
				}
				source = destroyChildren(source + source->_parentIndex);
				dest = _nodes.begin() + destIndex;
			}
			if (!canHoldLeafs(dest, +1)) {
				std::size_t sourceIndex = source - _nodes.begin();
				if (source > dest) {
					sourceIndex += (1 << Dim);
				}
				dest = createChildren(dest);
				dest = find(dest, position);
				source = _nodes.begin() + sourceIndex;
			}
		}
		return std::make_tuple(source, dest, moveAt(source, dest, leaf));
	}
	
	std::tuple<NodeIterator, NodeIterator, LeafIterator> move(
			LeafIterator leaf,
			Vector<Dim> const& position) {
		return move(root(), leaf, position);
	}
	///@}
	
	
	
	/**
	 * \brief Checks whether a Node contains a particular point.
	 */
	bool contains(ConstNodeIterator node, Vector<Dim> const& point) {
		Vector<Dim> lower = node->_position;
		Vector<Dim> upper = node->_position + node->_dimensions;
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			// De Morgan's law has been applied here so that NaN is dealt with
			// appropriately.
			if (!(point[dim] >= lower[dim] && point[dim] < upper[dim])) {
				return false;
			}
		}
		return true;
	}
	
	
	
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
			Vector<Dim> const& position) {
		bool contains = contains(start, position);
		auto node = start;
		if (contains && !node->_hasChildren) {
			return node;
		}
		else if (contains && node->_hasChildren) {
			auto child = node + 1;
			Vector<Dim> midpoint = node->_position + node->_dimensions / 2;
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				if (position[dim] >= midpoint[dim]) {
					child += (1 << dim);
				}
			}
			return find(child, position);
		}
		else if (node->_hasParent) {
			auto parent = node + node->_parentIndex;
			return find(parent, position);
		}
		return _nodes.end();
	}
	
	NodeIterator find(
			Vector<Dim> const& position) {
		return find(root(), position);
	}
	
	ConstNodeIterator find(
			ConstNodeIterator start,
			Vector<Dim> const& position) const {
		return const_cast<Octree<L, N, Dim>*>(this)->find(
			start, position);
	}
	
	ConstNodeIterator find(
			Vector<Dim> const& position) const {
		return find(root(), position);
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
			ConstLeafIterator leaf) {
		auto node = start;
		
		std::size_t leafIndex = leaf - _leafs.begin();
		std::size_t lower = node->_leafIndex;
		std::size_t upper = lower + node->_leafCount;
		
		bool contains = leafIndex >= lower && leafIndex < upper;
		
		if (contains && !node->_hasChildren) {
			return node;
		}
		else if (contains && node->_hasChildren) {
			for (std::size_t index = 0; index < (1 << Dim); ++index) {
				auto child = node + node->_childIndices[index];
				std::size_t lower = child->_leafIndex;
				std::size_t upper = lower + child->_leafCount;
				if (leafIndex >= lower && leafIndex < upper) {
					return find(child, leaf);
				}
			}
		}
		else if (node->_hasParent) {
			auto parent = node + node->_parentIndex;
			return find(parent, leaf);
		}
		return _nodes.end();
	}
	
	NodeIterator find(
			ConstLeafIterator leaf) {
		return find(root(), leaf);
	}
	
	ConstNodeIterator find(
			ConstNodeIterator start,
			ConstLeafIterator leaf) const {
		return const_cast<Octree<L, N, Dim>*>(this)->find(
			start, leaf);
	}
	
	ConstNodeIterator find(
			ConstLeafIterator leaf) const {
		return find(root(), leaf);
	}
	///@}
	
};

/**
 * \brief Represents a piece of data stored at the leaf of an Octree.
 * 
 * This class is a simple wrapper for the data as well as the position at which
 * the data is stored.
 */
template<typename L, typename N, std::size_t Dim>
class Octree<L, N, Dim>::Leaf final {
	
private:
	
	L _data;
	Vector<Dim> _position;
	
	Leaf(L data, Vector<Dim> position) :
			_data(data),
			_position(position) {
	}
	
public:
	
	///@{
	/**
	 * \brief Provides access to the (mutable) data contained at this leaf.
	 * \returns a reference to the data
	 */
	L& data() {
		return _data;
	}
	L const& data() const {
		return _data;
	}
	///@}
	
	/**
	 * \brief Gets the position of the leaf.
	 */
	Vector<Dim> const& position() const {
		return _position;
	}
	
};

/**
 * \brief Represents an internal node of an Octree.
 * 
 * This class is a simple wrapper for the data as well as the spatial region
 * encompassed by this node.
 */
template<typename L, typename N, std::size_t Dim>
class Octree<L, N, Dim>::Node final {
	
private:
	
	// The depth of this node within the octree (0 for root, and so on).
	std::size_t _depth;
	
	// Whether this node has any children.
	bool _hasChildren;
	// The indices of the children of this node, stored relative to the index
	// of this node. The last entry points to the next sibling of this node,
	// and is used to determine the total size of all of this node's children.
	std::size_t _childIndices[(1 << Dim) + 1];
	
	// Whether this node has a parent.
	bool _hasParent;
	// The relative index of the parent of this node.
	std::ptrdiff_t _parentIndex;
	// Which child # of its parent this node is. (0th child, 1st child, etc).
	std::size_t _siblingIndex;
	
	// The number of leaves that this node contains. This includes leaves stored
	// by all descendants of this node.
	std::size_t _leafCount;
	// The index within the octree's leaf array that this node's leaves are
	// located at.
	std::size_t _leafIndex;
	
	// The section of space that this node encompasses.
	Vector<Dim> _position;
	Vector<Dim> _dimensions;
	
	// The data stored at the node itself.
	N _data;
	
	Node() :
			_depth(0),
			_hasChildren(false),
			_childIndices(),
			_hasParent(false),
			_parentIndex(),
			_siblingIndex(),
			_leafCount(0),
			_leafIndex(0),
			_position(),
			_dimensions(),
			_data() {
	}
	
public:
	
	///@{
	/**
	 * \brief Provides access to the (mutable) data contained at this node.
	 * \returns a reference to the data
	 */
	N& data() {
		return _data;
	}
	N const& data() const {
		return _data;
	}
	///@}
	
	/**
	 * \brief Gets the position of the "upper-left" corner of the node.
	 * 
	 * This method will always return the corner that has the smallest coordinate
	 * values. Any point contained within this node is guaranteed to have
	 * coordinate values larger or equal to this point.
	 */
	Vector<Dim> const& position() const {
		return _position;
	}
	
	/**
	 * \brief Gets a Vector representing the size of the node.
	 * 
	 * Note that a node is not required to be a hyper-cube: it can (in general)
	 * be any hyper-rectangle.
	 */
	Vector<Dim> const& dimensions() const {
		return _dimensions;
	}
	
};

}

#endif

