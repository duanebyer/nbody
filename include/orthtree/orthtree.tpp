#ifndef __NBODY_ORTHTREE_ORTHTREE_TPP_
#define __NBODY_ORTHTREE_ORTHTREE_TPP_

#include "orthtree/orthtree.h"

#include "orthtree/iterator.h"
#include "orthtree/range.h"
#include "orthtree/reference.h"
#include "orthtree/value.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

namespace nbody {

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
Orthtree<Dim, Vector, LeafValue, NodeValue>::Orthtree(
		Vector position,
		Vector dimensions,
		typename LeafList::size_type nodeCapacity,
		typename NodeList::size_type maxDepth,
		bool adjust) :
		_leafs(),
		_nodes(),
		_nodeCapacity(nodeCapacity),
		_maxDepth(maxDepth),
		_adjust(adjust) {
	NodeInternal root(position, dimensions);
	_nodes.insert(_nodes.begin(), root);
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIterator
Orthtree<Dim, Vector, LeafValue, NodeValue>::createChildren(
		NodeIterator node) {
	// Create the 2^Dim child nodes inside the list of nodes. This will not
	// invalidate the node iterator.
	_nodes.insert(
		node.internalIt() + 1,
		1 << Dim,
		NodeInternal(node->position, node->dimensions));
	
	// Loop through the new children, and set up their various properties.
	for (std::size_t index = 0; index < (1 << Dim); ++index) {
		NodeInternal& child = *(node.internalIt() + index + 1);
		child.depth = node->depth + 1;
		child.hasParent = true;
		child.parentIndex = -((typename NodeRange::difference_type) index + 1);
		child.siblingIndex = index;
		child.leafIndex =
			node.internalIt()->leafIndex +
			node.internalIt()->leafCount;
		// Position and size the child node.
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			child.dimensions[dim] = child.dimensions[dim] / 2;
			if ((1 << dim) & index) {
				child.position[dim] =
					child.position[dim] + node->dimensions[dim] / 2;
			}
		}
		
		// Add the child to this node.
		node.internalIt()->childIndices[index] = index + 1;
	}
	node.internalIt()->childIndices[1 << Dim] = (1 << Dim) + 1;
	node.internalIt()->hasChildren = true;
	
	// Go through the parent, grandparent, great-grandparent, ...  of this
	// node and update their child indices.
	NodeIterator parent = node;
	while (parent->hasParent) {
		typename NodeList::size_type siblingIndex;
		siblingIndex = parent.internalIt()->siblingIndex;
		parent = parent->parent;
		while (++siblingIndex < (1 << Dim)) {
			parent.internalIt()->childIndices[siblingIndex] += (1 << Dim);
			NodeIterator child = parent->children[siblingIndex];
			child.internalIt()->parentIndex -= (1 << Dim);
		}
		parent.internalIt()->childIndices[1 << Dim] += (1 << Dim);
	}
	
	// Distribute the leaves of this node to the children.
	for (
			typename LeafList::size_type index = 0;
			index < node->leafs.size();
			++index) {
		typename NodeList::size_type childIndex = 0;
		// The leaf is always taken from the front of the list, and moved to
		// a location on the end of the list.
		LeafIterator leaf = node->leafs.begin();
		Vector const& position = leaf->position;
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (
					node->position[dim] + node->dimensions[dim] / 2 <
					position[dim]) {
				childIndex += (1 << dim);
			}
		}
		moveAt(node, node->children[childIndex], leaf);
	}
	
	return node;
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIterator
Orthtree<Dim, Vector, LeafValue, NodeValue>::destroyChildren(
		NodeIterator node) {
	// Determine how many children, grandchildren, great-grandchildren, ...
	// of this node.
	typename NodeList::size_type numDescendants = descendants(node).size();
	
	// Destroy the subnodes. This won't invalidate the node iterator.
	_nodes.erase(
		node->children[0].internalIt(),
		node->children[1 << Dim].internalIt());
	node.internalIt()->hasChildren = false;
	
	// Go through the parent, grandparent, great-grandparent, ... of this
	// node and update their child indices.
	NodeIterator parent = node;
	while (parent->hasParent) {
		typename NodeList::size_type siblingIndex;
		siblingIndex = parent.internalIt()->siblingIndex;
		parent = parent->parent;
		while (++siblingIndex < (1 << Dim)) {
			parent.internalIt()->childIndices[siblingIndex] -= numDescendants;
			NodeIterator child = parent->children[siblingIndex];
			child.internalIt()->parentIndex += numDescendants;
		}
		parent.internalIt()->childIndices[1 << Dim] -= numDescendants;
	}
	
	return node;
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafIterator
Orthtree<Dim, Vector, LeafValue, NodeValue>::insertAt(
		NodeIterator node,
		LeafValue const& value,
		Vector const& position) {
	// Add the leaf to the master list of leaves in the orthtree.
	_leafs.insert(
		node->leafs.end().internalIt(),
		LeafInternal(value, position));
	
	// Loop through the rest of the nodes and increment their leaf indices
	// so that they still refer to the correct location in the leaf vector.
	auto currentNode = node.internalIt();
	while (++currentNode != _nodes.end()) {
		++currentNode->leafIndex;
	}
	
	// Also loop through all ancestors and increment their leaf counts.
	NodeIterator parent = node;
	bool hasParent = true;
	while (hasParent) {
		++parent.internalIt()->leafCount;
		hasParent = parent->hasParent;
		parent = parent->parent;
	}
	
	LeafIterator result = node->leafs.end(); --result;
	return result;
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafIterator
Orthtree<Dim, Vector, LeafValue, NodeValue>::eraseAt(
		NodeIterator node,
		LeafIterator leaf) {
	// Remove the leaf from the master orthtree leaf vector.
	_leafs.erase(leaf.internalIt());
	
	// Loop through the rest of the nodes and increment their leaf indices
	// so that they still refer to the correct location in the leaf vector.
	auto currentNode = node.internalIt();
	while (++currentNode != _nodes.end()) {
		--currentNode->leafIndex;
	}
	
	// Loop through all of the ancestors of this node and decremement their
	// leaf counts.
	NodeIterator parent = node;
	bool hasParent = true;
	while (hasParent) {
		--parent.internalIt()->leafCount;
		hasParent = parent->hasParent;
		parent = parent->parent;
	}
	
	return leaf;
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafIterator
Orthtree<Dim, Vector, LeafValue, NodeValue>::moveAt(
		NodeIterator sourceNode,
		NodeIterator destNode,
		LeafIterator sourceLeaf) {
	// Determine the relative ordering of the source/destination leafs.
	LeafIterator destLeaf = destNode->leafs.end(); --destLeaf;
	bool inverted = sourceLeaf._index > destLeaf._index;
	LeafIterator firstLeaf = inverted ? destLeaf : sourceLeaf;
	LeafIterator lastLeaf = inverted ? sourceLeaf : destLeaf;
	
	// Move the leaf from its old position to its new position.
	std::rotate(
		firstLeaf.internalIt() + inverted,
		sourceLeaf.internalIt() + !inverted,
		lastLeaf.internalIt() + 1);
	
	// Adjust the ancestors of the source node.
	NodeIterator sourceParentNode = sourceNode;
	while (!contains(sourceParentNode, destLeaf)) {
		--sourceParentNode.internalIt()->leafCount;
		sourceParentNode = sourceParentNode->parent;
	}
	
	// Adjust the ancestors of the destination node.
	NodeIterator destParentNode = destNode;
	while (!contains(destParentNode, sourceLeaf)) {
		++destParentNode.internalIt()->leafCount;
		destParentNode = destParentNode->parent;
	}
	
	// Both sourceParentNode and destParentNode should be equal to the same
	// node: the most recent common ancestor of both the source and
	// destination.
	
	// Adjust the nodes in between the source and destination node.
	signed char indexOffset = inverted ? +1 : -1;
	NodeIterator firstNode = inverted ? destNode : sourceNode;
	NodeIterator lastNode = inverted ? sourceNode : destNode;
	++firstNode;
	++lastNode;
	for (NodeIterator node = firstNode; node != lastNode; ++node) {
		node.internalIt()->leafIndex += indexOffset;
	}
	
	return destLeaf;
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafRange
Orthtree<Dim, Vector, LeafValue, NodeValue>::leafs() {
	return LeafRange(this, 0, _leafs.size());
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::ConstLeafRange
Orthtree<Dim, Vector, LeafValue, NodeValue>::leafs() const {
	return ConstLeafRange(this, 0, _leafs.size());
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::ConstLeafRange
Orthtree<Dim, Vector, LeafValue, NodeValue>::cleafs() const {
	return ConstLeafRange(this, 0, _leafs.size());
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeRange
Orthtree<Dim, Vector, LeafValue, NodeValue>::nodes() {
	return NodeRange(this, 0, _nodes.size());
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::ConstNodeRange
Orthtree<Dim, Vector, LeafValue, NodeValue>::nodes() const {
	return ConstNodeRange(this, 0, _nodes.size());
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::ConstNodeRange
Orthtree<Dim, Vector, LeafValue, NodeValue>::cnodes() const {
	return ConstNodeRange(this, 0, _nodes.size());
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIterator
Orthtree<Dim, Vector, LeafValue, NodeValue>::root() {
	return NodeIterator(this, 0);
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::ConstNodeIterator
Orthtree<Dim, Vector, LeafValue, NodeValue>::root() const {
	return ConstNodeIterator(this, 0);
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::ConstNodeIterator
Orthtree<Dim, Vector, LeafValue, NodeValue>::croot() const {
	return ConstNodeIterator(this, 0);
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeRange
Orthtree<Dim, Vector, LeafValue, NodeValue>::descendants(
		ConstNodeIterator node) {
	return NodeRange(
		this,
		node->children[0]._index,
		node->children[1 << Dim]._index);
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::ConstNodeRange
Orthtree<Dim, Vector, LeafValue, NodeValue>::descendants(
		ConstNodeIterator node) const {
	return ConstNodeRange(
		this,
		node->children[0]._index,
		node->children[1 << Dim]._index);
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::ConstNodeRange
Orthtree<Dim, Vector, LeafValue, NodeValue>::cdescendants(
		ConstNodeIterator node) const {
	return ConstNodeRange(
		this,
		node->children[0]._index,
		node->children[1 << Dim]._index);
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
bool Orthtree<Dim, Vector, LeafValue, NodeValue>::adjust(
		ConstNodeIterator node) {
	bool result = false;
	// If the node doesn't have children but should, then make them.
	if (!node->hasChildren && !canHoldLeafs(node, 0)) {
		node = createChildren(node);
		result = true;
	}
	// If the node does have children but shouldn't, then remove them.
	else if (node->hasChildren && canHoldLeafs(node, 0)) {
		node = destroyChildren(node);
		result = true;
	}
	// Then, adjust all of this node's children as well.
	if (node->hasChildren) {
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
			ConstNodeIterator child = node->children[index];
			bool nextResult = adjust(child);
			result = result || nextResult;
		}
	}
	return result;
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
std::tuple<
		typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIterator,
		typename Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafIterator>
Orthtree<Dim, Vector, LeafValue, NodeValue>::insert(
		ConstNodeIterator hint,
		LeafValue const& value,
		Vector const& position) {
	// Find the node with the correct position, and insert the leaf into
	// that node.
	NodeIterator node = find(hint, position);
	if (node == nodes().end()) {
		return std::make_tuple(nodes().end(), leafs().end());
	}
	// Create children if the node doesn't have the capacity to store
	// this leaf.
	while (_adjust && !canHoldLeafs(node, +1)) {
		node = createChildren(node);
		node = find(node, position);
	}
	return std::make_tuple(node, insertAt(node, value, position));
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
std::tuple<
		typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIterator,
		typename Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafIterator>
Orthtree<Dim, Vector, LeafValue, NodeValue>::erase(
		ConstNodeIterator hint,
		LeafIterator leaf) {
	// Find the node that contains this leaf, and then erase the leaf from
	// that node.
	NodeIterator node = find(hint, leaf);
	if (node == nodes().end()) {
		return std::make_tuple(nodes().end(), leafs().end());
	}
	// If the parent of this node doesn't need to be divided into subnodes
	// anymore, then merge its children together.
	while (
			_adjust &&
			node->hasParent &&
			canHoldLeafs(node->parent, -1)) {
		node = destroyChildren(node->parent);
	}
	return std::make_tuple(node, eraseAt(node, leaf));
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
std::tuple<
		typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIterator,
		typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIterator,
		typename Orthtree<Dim, Vector, LeafValue, NodeValue>::LeafIterator>
Orthtree<Dim, Vector, LeafValue, NodeValue>::move(
		ConstNodeIterator hint,
		LeafIterator leaf,
		Vector const& position) {
	// Find the source node that contains the leaf, and the target node with
	// the correct position.
	NodeIterator source = find(hint, leaf);
	NodeIterator dest = find(hint, position);
	if (source == nodes().end() || dest == nodes().end()) {
		return std::make_tuple(nodes().end(), nodes().end(), leafs().end());
	}
	// If the source and the destination are distinct, then check to make
	// sure that they remain within the node capcity.
	if (_adjust && source != dest) {
		while (
				source->hasParent &&
				canHoldLeafs(source->parent, -1)) {
			if (dest > source) {
				dest -= (1 << Dim);
			}
			source = destroyChildren(source->parent);
		}
		if (!canHoldLeafs(dest, +1)) {
			if (source > dest) {
				source += (1 << Dim);
			}
			dest = createChildren(dest);
			dest = find(dest, position);
		}
	}
	return std::make_tuple(source, dest, moveAt(source, dest, leaf));
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIterator
Orthtree<Dim, Vector, LeafValue, NodeValue>::find(
		ConstNodeIterator hint,
		Vector const& point) {
	// If the hint node doesn't contain the point, then go up the tree until we
	// reach a node that does contain the point.
	NodeIterator node(this, hint._index);
	while (!contains(node, point)) {
		if (node->hasParent) {
			node = node->parent;
		}
		else {
			return nodes().end();
		}
	}
	
	// Then, go down the tree until we reach the deepest node that contains the
	// point.
	while (node->hasChildren) {
		node = findChild(node, point);
	}
	
	return node;
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIterator
Orthtree<Dim, Vector, LeafValue, NodeValue>::find(
		ConstNodeIterator hint,
		ConstLeafIterator leaf) {
	// If the hint node doesn't contain the leaf, then go up the tree until we
	// reach a node that does contain the leaf.
	NodeIterator node(this, hint._index);
	while (!contains(node, leaf)) {
		if (node->hasParent) {
			node = node->parent;
		}
		else {
			return nodes().end();
		}
	}
	
	// Then go down the tree until we reach the deepest node that contains the
	// point.
	while (node->hasChildren) {
		node = findChild(node, leaf);
	}
	
	return node;
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIterator
Orthtree<Dim, Vector, LeafValue, NodeValue>::findChild(
		ConstNodeIterator node,
		Vector const& point) {
	typename NodeList::size_type childIndex = 0;
	for (std::size_t dim = 0; dim < Dim; ++dim) {
		if (point[dim] >= node->position[dim] + node->dimensions[dim] / 2) {
			childIndex += (1 << dim);
		}
	}
	ConstNodeIterator child = node->children[childIndex];
	return NodeIterator(this, child._index);
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
typename Orthtree<Dim, Vector, LeafValue, NodeValue>::NodeIterator
Orthtree<Dim, Vector, LeafValue, NodeValue>::findChild(
		ConstNodeIterator node,
		ConstLeafIterator leaf) {
	for (
			typename NodeList::size_type childIndex = 0;
			childIndex < (1 << Dim);
			++childIndex) {
		ConstNodeIterator child = node->children[childIndex];
		if (contains(child, leaf)) {
			return NodeIterator(this, child._index);
		}
	}
	return nodes().end();
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
bool Orthtree<Dim, Vector, LeafValue, NodeValue>::contains(
		ConstNodeIterator node,
		Vector const& point) const {
	for (std::size_t dim = 0; dim < Dim; ++dim) {
		Scalar lower = node->position[dim];
		Scalar upper = lower + node->dimensions[dim];
		if (!(point[dim] >= lower && point[dim] < upper)) {
			return false;
		}
	}
	return true;
}

template<
	std::size_t Dim,
	typename Vector,
	typename LeafValue,
	typename NodeValue>
bool Orthtree<Dim, Vector, LeafValue, NodeValue>::contains(
		ConstNodeIterator node,
		ConstLeafIterator leaf) const {
	using Index = typename LeafList::size_type;
	Index leafIndex = (Index) leaf._index;
	Index lower = node.internalIt()->leafIndex;
	Index upper = lower + node.internalIt()->leafCount;
	return leafIndex >= lower && leafIndex < upper;
}

}

#endif

