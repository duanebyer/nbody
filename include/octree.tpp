#ifndef __NBODY_OCTREE_TPP_
#define __NBODY_OCTREE_TPP_

#include <algorithm>
#include <climits>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

using namespace nbody;

// *----------------*
// | Octree methods |
// *----------------*

template<typename L, typename N, std::size_t Dim>
Octree<L, N, Dim>::Octree(
		Vector<Dim> position,
		Vector<Dim> dimensions,
		LeafList::size_type nodeCapacity,
		NodeList::size_type maxDepth,
		bool adjust) :
		_leafs(),
		_nodes(),
		_nodeCapacity(nodeCapacity),
		_maxDepth(maxDepth),
		_adjust(adjust) {
	NodeInternal root;
	root.position = position;
	root.dimensions = dimensions;
	_nodes.insert(_nodes.begin(), root);
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeIterator Octree<L, N, Dim>::createChildren(
		NodeIterator node) {
	// Create the 2^Dim child nodes inside the list of nodes. This will not
	// invalidate the node iterator.
	_nodes.insert(node.internalIt() + 1, 1 << Dim, NodeInternal());
	
	// Loop through the new children, and set up their various properties.
	Vector<Dim> childDimensions = node->dimensions / 2;
	Vector<Dim> midpoint = node->position + childDimensions;
	for (std::size_t index = 0; index < (1 << Dim); ++index) {
		NodeInternal& child = *(node.internalIt() + index + 1);
		child.depth = node->depth + 1;
		child.hasParent = true;
		child.parentIndex = -((typename NodeRange::difference_type) index + 1);
		child.siblingIndex = index;
		child.leafIndex =
			node.internalIt()->leafIndex +
			node.internalIt()->leafCount;
		child.dimensions = childDimensions;
		child.position = node->position;
		// Shift the child position depending on which child it is.
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if ((1 << dim) & index) {
				child.position[dim] = midpoint[dim];
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
	while (parent.hasParent()) {
		NodeList::size_type siblingIndex = parent.internalIt()->siblingIndex;
		parent = parent.parent();
		while (++siblingIndex < (1 << Dim)) {
			parent.internalIt()->childIndices[siblingIndex] += (1 << Dim);
			NodeIterator child = parent->children[siblingIndex];
			child.internalIt()->parentIndex -= (1 << Dim);
		}
		parent.internalIt()->childIndices[1 << Dim] += (1 << Dim);
	}
	
	// Distribute the leaves of this node to the children.
	for (LeafList::size_type index = 0; index < node.leafs().size(); ++index) {
		// Figure out which node the leaf belongs to.
		NodeList::size_type childIndex = 0;
		LeafIterator leaf = node.leafs().begin();
		Vector<Dim> const& position = leaf.position();
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (position[dim] >= midpoint[dim]) {
				childIndex += (1 << dim);
			}
		}
		moveAt(node, node->children[childIndex], leaf);
	}
	
	return node;
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeIterator Octree<L, N, Dim>::destroyChildren(
		NodeIterator node) {
	// Determine how many children, grandchildren, great-grandchildren, ...
	// of this node.
	NodeList::size_type numDescendants = descendants(node).size();
	
	// Destroy the subnodes. This won't invalidate the node iterator.
	_nodes.erase(
		node->children[0].internalIt(),
		node->children[1 << Dim].internalIt());
	node.internalIt()->hasChildren = false;
	
	// Go through the parent, grandparent, great-grandparent, ... of this
	// node and update their child indices.
	NodeIterator parent = node;
	while (parent->hasParent) {
		NodeList::size_type siblingIndex = parent.internalIt()->siblingIndex;
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

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::LeafIterator Octree<L, N, Dim>::insertAt(
		NodeIterator node,
		L const& value,
		Vector<Dim> const& position) {
	// Add the leaf to the master list of leaves in the octree.
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
	
	return node->leafs.end() - 1;
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::LeafIterator Octree<L, N, Dim>::eraseAt(
		NodeIterator node,
		LeafIterator leaf) {
	// Remove the leaf from the master octree leaf vector.
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

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::LeafIterator Octree<L, N, Dim>::moveAt(
		NodeIterator sourceNode,
		NodeIterator destNode,
		LeafIterator sourceLeaf) {
	// Reinsert the leaf into the leaf vector in its new position.
	LeafIterator destLeaf = destNode->leafs.end() - 1;
	bool inverted = sourceLeaf > destLeaf;
	LeafIterator firstLeaf = inverted ? destLeaf : sourceLeaf;
	LeafIterator lastLeaf = inverted ? sourceLeaf : destLeaf;
	
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
	std::ptrdiff_t invertedSign = inverted ? -1 : +1;
	for (NodeIterator node = sourceNode + 1; node <= destNode; ++node) {
		node.internalIt()->leafIndex -= invertedSign;
	}
	
	return destLeaf;
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::LeafRange Octree<L, N, Dim>::leafs() {
	return LeafRange(this, 0, _leafs.size());
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::ConstLeafRange Octree<L, N, Dim>::leafs() const {
	return ConstLeafRange(this, 0, _leafs.size());
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::ConstLeafRange Octree<L, N, Dim>::cleafs() const {
	return ConstLeafRange(this, 0, _leafs.size());
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeRange Octree<L, N, Dim>::nodes() {
	return NodeRange(this, 0, _nodes.size());
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::ConstNodeRange Octree<L, N, Dim>::nodes() const {
	return ConstNodeRange(this, 0, _nodes.size());
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::ConstNodeRange Octree<L, N, Dim>::cnodes() const {
	return ConstNodeRange(this, 0, _nodes.size());
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeIterator Octree<L, N, Dim>::root() {
	return NodeIterator(this, 0);
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::ConstNodeIterator Octree<L, N, Dim>::root() const {
	return ConstNodeIterator(this, 0);
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::ConstNodeIterator Octree<L, N, Dim>::croot() const {
	return ConstNodeIterator(this, 0);
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeRange Octree<L, N, Dim>::descendants(
		NodeIterator node) {
	return NodeRange(
		this,
		node->children[0]._index,
		node->children[1 << Dim]._index);
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::ConstNodeRange descendants(
		ConstNodeIterator node) {
	return ConstNodeRange(
		this,
		node->children[0]._index,
		node->children[1 << Dim]._index);
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::ConstNodeRange descendants(
		ConstNodeIterator node) const {
	return ConstNodeRange(
		this,
		node->children[0]._index,
		node->children[1 << Dim]._index);
}

template<typename L, typename N, std::size_t Dim>
bool Octree<L, N, Dim>::adjust(
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

template<typename L, typename N, std::size_t Dim>
std::tuple<
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::insert(
		ConstNodeIterator hint,
		L const& data,
		Vector<Dim> const& position) {
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

template<typename L, typename N, std::size_t Dim>
std::tuple<
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::insert(
		L const& data,
		Vector<Dim> const& position) {
	return insert(root(), data, position);
}

template<typename L, typename N, std::size_t Dim>
template<typename LeafTuple>
std::tuple<
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::insert(
		ConstNodeIterator hint,
		LeafTuple leafPair) {
	return insert(
		hint,
		std::get<L>(leafPair),
		std::get<Vector<Dim> >(leafPair));
}

template<typename L, typename N, std::size_t Dim>
template<typename LeafTuple>
std::tuple<
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::insert(
		LeafTuple leafPair) {
	return insert(
		root(),
		std::get<L>(leafPair),
		std::get<Vector<Dim> >(leafPair));
}

template<typename L, typename N, std::size_t Dim>
std::tuple<
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::erase(
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

template<typename L, typename N, std::size_t Dim>
std::tuple<
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::erase(
		LeafIterator leaf) {
	return erase(root(), leaf);
}

template<typename L, typename N, std::size_t Dim>
std::tuple<
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::move(
		ConstNodeIterator hint,
		LeafIterator leaf,
		Vector<Dim> const& position) {
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

template<typename L, typename N, std::size_t Dim>
std::tuple<
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::move(
		LeafIterator leaf,
		Vector<Dim> const& position) {
	return move(root(), leaf, position);
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeIterator Octree<L, N, Dim>::find(
		ConstNodeIterator hint,
		Vector<Dim> const& position) {
	bool contains = hint.contains(position);
	// If this node is childless and contains the point, then just return it.
	if (contains && !hint->hasChildren) {
		return NodeIterator(this, hint._index);
	}
	// If it is childless but contains the point, then recursively call this
	// method on the child that also conains the point.
	else if (contains && hint->hasChildren) {
		return find(findChild(hint, position), position);
	}
	// If it does not contain the point, then recursively call this method on
	// the parent of this node.
	else if (hint->hasParent) {
		return find(hint->parent, position);
	}
	return nodes().end();
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeIterator Octree<L, N, Dim>::find(
		ConstNodeIterator hint,
		ConstLeafIterator leaf) {
	bool contains = hint.contains(leaf);
	// If this node is childless and contains the leaf, then return itself.
	if (contains && !hint->hasChildren) {
		return NodeIterator(this, hint._index);
	}
	// If it has children and contains the leaf, then recursively call this
	// method on the child that contains the leaf.
	else if (contains && hint->hasChildren) {
		return find(findChild(hint, leaf), leaf);
	}
	// If it doesn't contain the leaf, the recursively call this method on the
	// parent of this node.
	else if (hint->hasParent) {
		return find(hint->parent, leaf);
	}
	return nodes().end();
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeIterator Octree<L, N, Dim>::findChild(
		ConstNodeIterator node,
		Vector<Dim> const& point) {
	NodeList::size_type childIndex = 0;
	for (std::size_t dim = 0; dim < Dim; ++dim) {
		if (point[dim] >= node->position[dim] + node->dimensions[dim] / 2.0) {
			childIndex += (1 << dim);
		}
	}
	ConstNodeIterator child = node->children[childIndex];
	return NodeIterator(this, child._index);
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeIterator Octree<L, N, Dim>::findChild(
		ConstNodeIterator node,
		ConstLeafIterator leaf) {
	for (
			NodeList::size_type childIndex = 0;
			childIndex < (1 << Dim);
			++childIndex) {
		ConstNodeIterator child = node->children[childIndex];
		if (contains(child, leaf)) {
			return NodeIterator(this, child._index);
		}
	}
	return nodes().end();
}

#endif

