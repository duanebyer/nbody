#ifndef __NBODY_OCTREE_H_
#define __NBODY_OCTREE_H_

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "tensor.h"

namespace nbody {

/**
 * \brief Wraps a piece of data together with a position so that it can be
 * stored at the leaf of an Octree.
 * 
 * \tparam Leaf the type of data stored at the leaves of the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename Leaf, std::size_t Dim>
class OctreeLeaf final {
	
	template<typename Leaf, typename Node, std::size_t Dim>
	friend class Octree;
	
private:
	
	Leaf _data;
	Vector<Dim> _position;
	
	OctreeLeaf(Leaf data, Vector<Dim> position) :
			_data(data),
			_position(position) {
	}
	
public:
	
	Leaf& data() {
		return _data;
	}
	Leaf const& data() const {
		return _data;
	}
	
	Vector<Dim> const& position() const {
		return _position;
	}
	
};

/**
 * \brief Represents an Octree node containing child nodes and possibly leaves.
 * 
 * Each OctreeNode may contain a certain number of children (the number of
 * children depends upon the dimension of the space).
 * 
 * In addition, each OctreeNode has node data associated with it, regardless of
 * whether it is a leaf or not. This data could contain, for example, the
 * center of mass of all of the children of the OctreeNode. Every time a
 * change is made to the OctreeNode or its children, its internal data is also
 * updated.
 * 
 * \tparam Node the type of data stored at the nodes of the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename Node, std::size_t Dim>
class OctreeNode final {
	
	template<typename Leaf, typename Node, std::size_t Dim>
	friend class Octree;
	
private:
	
	// Whether this node has any children currently.
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
	// The leaves that this node contains are located at this index.
	std::size_t _leafIndex;
	
	// The section of space that this node encompasses.
	Vector<Dim> _position;
	Vector<Dim> _dimensions;
	
	// The data stored at the node itself.
	Node _data;
	
	OctreeNode() :
			_hasChildren(false),
			_childIndices(),
			_hasParent(false),
			_leafCount(0),
			_leafIndex(0),
			_position(),
			_dimensions() {
	}
	
public:
	
	Node& data() {
		return _data;
	}
	Node const& data() const {
		return _data;
	}
	
	Vector<Dim> const& position() const {
		return _position;
	}
	Vector<Dim> const& dimensions() const {
		return _dimensions;
	}
	
};

/**
 * \brief A data structure that stores positional data in arbitrary
 * dimensional space.
 * 
 * \tparam Leaf the type of data stored at the leaves of the Octree
 * \tparam Node the type of data stored at the nodes of the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename Leaf, typename Node, std::size_t Dim>
class Octree final {
	
private:
	
	// A list storing all of the nodes of the octree.
	std::vector<OctreeNode<Node, Dim> > _nodes;
	
	// A list storing all of the leaves of the octree (it's spelled wrong for
	// consistancy).
	std::vector<OctreeLeaf<Leaf, Dim> > _leafs;
	
	// The number of leaves to store at a single node of the octree.
	std::size_t _nodeCapacity;
	

	// Divides a node into a set of subnodes and partitions its leaves between
	// them.
	void createChildren(ConstNodeIterator node) {
		// Create the 2^Dim child nodes inside the parent octree.
		auto firstChild = _nodes.insert(
			node, 1 << Dim,
			OctreeNode<Leaf, Node, Dim>());
		
		// Update the node iterator as it has been invalidated.
		node = firstChild - 1;
		
		// Loop through the new children, and set up their various properties.
		Vector<Dim> childDimensions = _dimensions / 2;
		Vector<Dim> midpoint = _position + childDimensions;
		auto child = firstChild;
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
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
			std::childIndex = 0;
			auto leaf = _leafs.begin() + node->_leafIndex + index;
			Vector<Dim> const& position = leaf->position();
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				if (position[dim] >= midpoint[dim]) {
					childIndex += (1 << dim);
				}
			}
			moveAt(node, firstChild + childIndex, leaf);
		}
	}
	
	// Destroys all descendants of a node and takes their leaves into the node.
	void destroyChildren(ConstNodeIterator node) {
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
	}
	
	// Adds a leaf to a specific node.
	LeafIterator insertAt(
			ConstNodeIterator node,
			OctreeLeaf<Leaf, Dim> const& leaf) {
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
		
		std::size_t sourceLeafIndex = sourceLeaf - beginLeaf;
		std::size_t destLeafIndex = destLeaf - beginLeaf;
		
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
		
		return beginLeaf + destLeafIndex;
	}
	
public:
	
	using NodeIterator = decltype(_nodes)::iterator;
	using ConstNodeIterator = decltype(_nodes)::const_iterator;
	using LeafIterator = decltype(_leafs)::iterator;
	using ConstLeafIterator = decltype(_leafs)::const_iterator;
	
	
	
	Octree(Vector<Dim> position, Vector<Dim> dimensions) :
			_nodes(),
			_leafs(),
			_nodeCapacity(1) {
	}
	
	
	
	NodeIterator root() {
		return _nodes.begin();
	}
	ConstNodeIterator root() const {
		return _nodes.begin();
	}
	
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
	
	NodeIterator parent(ConstNodeIterator node) {
		if (!node->_hasParent) {
			return _nodes.end();
		}
		else {
			return node + node->_parentIndex;
		}
	}
	ConstNodeIterator parent(ConstNodeIterator node) const {
		return const_cast<Octree<Leaf, Node, Dim>*>(this)->parent(node);
	}
	NodeIterator child(ConstNodeIterator node, std::size_t index) {
		if (!node->_hasChildren) {
			return _nodes.end();
		}
		else {
			return node + node->_childIndices[index];
		}
	}
	ConstNodeIterator child(ConstNodeIterator node, std::size_t index) const {
		return const_cast<Octree<Leaf, Node, Dim>*>(this)->child(node, index);
	}
	
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
	
	
	
	std::tuple<NodeIterator, LeafIterator> insert(
			ConstNodeIterator start,
			OctreeLeaf<Leaf, Dim> const& leaf) {
		// Find the node with the correct position, and insert the leaf into
		// that node.
		auto node = find(start, leaf.position());
		if (node == _nodes.end()) {
			return _leafs.end();
		}
		// Create children if the node doesn't have the capacity to store
		// this leaf.
		if (node->_leafCount + 1 >= _nodeCapacity) {
			createChildren(node);
			node = find(node, leaf.position());
		}
		return std::make_tuple(node, insertAt(node, leaf));
	}
	
	std::tuple<NodeIterator, LeafIterator> insert(
			OctreeLeaf<Leaf, Dim> const& leaf) {
		return insert(root(), leaf);
	}
	
	std::tuple<NodeIterator, LeafIterator> insert(
			ConstNodeIterator start,
			Leaf const& data,
			Vector<Dim> const& position) {
		OctreeLeaf<Leaf, Dim> leaf(data, position);
		return insert(start, leaf);
	}
	
	std::tuple<NodeIterator, LeafIterator> insert(
			Leaf const& data,
			Vector<Dim> const& position) {
		return insert(root(), data, position);
	}
	
	
	
	std::tuple<NodeIterator, LeafIterator> erase(
			ConstNodeIterator start,
			LeafIterator leaf) {
		// Find the node that contains this leaf, and then erase the leaf from
		// that node.
		auto node = find(start, leaf);
		if (node == _nodes.end()) {
			return _leafs.end();
		}
		return std::make_tuple(node, eraseAt(node, leaf));
	}
	
	std::tuple<NodeIterator, LeafIterator> erase(
			LeafIterator leaf) {
		return erase(root(), leaf);
	}
	
	
	
	std::tuple<NodeIterator, NodeIterator, LeafIterator> move(
			ConstNodeIterator start,
			LeafIterator leaf,
			Vector<Dim> const& position) {
		// Find the source node that contains the leaf, and the target node
		// with the correct position.
		auto source = find(start, leaf);
		auto dest = find(start, position);
		if (source == _nodes.end() || dest == _nodes.end()) {
			return _leafs.end();
		}
		// If the source and the destination are distinct, then check to make
		// sure that they remain within the node capcity.
		if (source != dest) {
			if (source->_hasParent) {
				auto parent = source + source->_parentIndex;
				if (parent->_leafCount - 1 < _nodeCapacity) {
					destroyChildren(parent);
					source = parent;
				}
			}
			if (dest->_leafCount + 1 >= _octree->_nodeCapacity) {
				createChildren(dest);
				dest = find(dest, position);
			}
		}
		return std::make_tuple(source, dest, moveAt(source, dest, leaf));
	}
	
	std::tuple<NodeIterator, NodeIterator, LeafIterator> move(
			LeafIterator leaf,
			Vector<Dim> const& position) {
		return move(root(), leaf, position);
	}
	
	
	
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
		return const_cast<Octree<Leaf, Node, Dim>*>(this)->find(
			start, position);
	}
	
	ConstNodeIterator find(
			Vector<Dim> const& position) const {
		return find(root(), position);
	}
	
	
	
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
		return const_cast<Octree<Leaf, Node, Dim>*>(this)->find(
			start, leaf);
	}
	
	ConstNodeIterator find(
			ConstLeafIterator leaf) const {
		return find(root(), leaf);
	}
	
};

}

#endif

