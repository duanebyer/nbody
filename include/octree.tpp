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
		std::size_t nodeCapacity,
		std::size_t maxDepth,
		bool adjust) :
		_leafs(),
		_nodes(),
		_nodeCapacity(nodeCapacity),
		_maxDepth(maxDepth),
		_adjust(adjust) {
	Node root;
	root.position = position;
	root.dimensions = dimensions;
	_nodes.insert(_nodes.begin(), root);
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeIterator Octree<L, N, Dim>::createChildren(
		ConstNodeIterator node) {
	// Create the 2^Dim child nodes inside the list of nodes. This will not
	// invalidate the node iterator.
	_nodes.insert(node.listIt() + 1, 1 << Dim, Node());
	
	// Loop through the new children, and set up their various properties.
	Vector<Dim> childDimensions = node.listRef().dimensions / 2;
	Vector<Dim> midpoint = node.listRef().position + childDimensions;
	for (std::size_t index = 0; index < (1 << Dim); ++index) {
		Node& child = *(node.listIt() + index + 1);
		child.depth = node.listRef().depth + 1;
		child.hasParent = true;
		child.parentIndex = -((typename NodeRange::difference_type) index + 1);
		child.siblingIndex = index;
		child.leafIndex = node.listRef().leafIndex + node.listRef().leafCount;
		child.dimensions = childDimensions;
		child.position = node.position();
		// Shift the child position depending on which child it is.
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if ((1 << dim) & index) {
				child.position[dim] = midpoint[dim];
			}
		}
		
		// Add the child to this node.
		node.listRef().childIndices[index] = index + 1;
	}
	node.listRef().hasChildren = true;
	
	// Go through the parent, grandparent, great-grandparent, ...  of this
	// node and update their child indices.
	ConstNodeIterator parent = node;
	while (parent.hasParent()) {
		std::size_t siblingIndex = parent.listRef().siblingIndex;
		parent = parent.parent();
		while (++siblingIndex <= (1 << Dim)) {
			parent.listRef().childIndices[siblingIndex] += (1 << Dim);
			ConstNodeIterator child = parent.child(siblingIndex);
			child.listRef().parentIndex -= (1 << Dim);
		}
	}
	
	// Distribute the leaves of this node to the children.
	for (std::size_t index = 0; index < node.leafs().size(); ++index) {
		// Figure out which node the leaf belongs to.
		std::size_t childIndex = 0;
		ConstLeafIterator leaf = node.leafs()[index];
		Vector<Dim> const& position = leaf.position();
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (position[dim] >= midpoint[dim]) {
				childIndex += (1 << dim);
			}
		}
		moveAt(node, node.child(childIndex), leaf);
	}
	
	return node;
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeIterator Octree<L, N, Dim>::destroyChildren(
		ConstNodeIterator node) {
	// Determine how many children, grandchildren, great-grandchildren, ...
	// of this node.
	std::size_t numDescendants = node.children().size();
	
	// Destroy the subnodes. This won't invalidate the node iterator.
	_nodes->erase(
		node.children().begin().listIt(),
		node.children().end().listIt());
	node.listRef().hasChildren = false;
	
	// Go through the parent, grandparent, great-grandparent, ... of this
	// node and update their child indices.
	ConstNodeIterator parent = node;
	while (parent.hasParent()) {
		std::size_t siblingIndex = parent.listRef().siblingIndex;
		parent = parent.parent();
		while (++siblingIndex <= (1 << Dim)) {
			parent.listRef().childIndices[siblingIndex] -= numDescendants;
			ConstNodeIterator child = parent.child(siblingIndex);
			child.listRef().parentIndex += numDescendants;
		}
	}
	
	return node;
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::LeafIterator Octree<L, N, Dim>::insertAt(
		ConstNodeIterator node,
		Leaf const& leaf) {
	// Add the leaf to the master list of leaves in the octree.
	_leafs.insert(node.leafs().end().listIt());
	
	// Loop through the rest of the nodes and increment their leaf indices
	// so that they still refer to the correct location in the leaf vector.
	auto currentNode = node.listIt();
	while (++currentNode != _nodes.end()) {
		++currentNode->leafIndex;
	}
	
	// Also loop through all ancestors and increment their leaf counts.
	ConstNodeIterator parent = node;
	while (parent.hasParent()) {
		++parent.listRef().leafCount;
		parent = parent.parent();
	}
	
	return node.leafs().end() - 1;
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::LeafIterator Octree<L, N, Dim>::eraseAt(
		ConstNodeIterator node,
		ConstLeafIterator leaf) {
	// Remove the leaf from the master octree leaf vector.
	_leafs.erase(leaf.listIt());
	
	// Loop through the rest of the nodes and increment their leaf indices
	// so that they still refer to the correct location in the leaf vector.
	auto currentNode = node.listIt();
	while (++currentNode != _nodes.end()) {
		--currentNode->leafIndex;
	}
	
	// Loop through all of the ancestors of this node and decremement their
	// leaf counts.
	ConstNodeIterator parent = node;
	while (parent.hasParent()) {
		--parent.listRef().leafCount;
		parent = parent.parent();
	}
	
	return leaf;
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::LeafIterator Octree<L, N, Dim>::moveAt(
		ConstNodeIterator sourceNode,
		ConstNodeIterator destNode,
		ConstLeafIterator sourceLeaf) {
	// Reinsert the leaf into the leaf vector in its new position.
	ConstLeafIterator destLeaf = destNode.leafs().end();
	bool inverted = sourceLeaf > destLeaf;
	ConstLeafIterator firstLeaf = inverted ? destLeaf : sourceLeaf;
	ConstLeafIterator lastLeaf = inverted ? sourceLeaf : destLeaf;
	
	std::rotate(
		firstLeaf.listIt(),
		sourceLeaf.listIt() + !inverted,
		lastLeaf.listIt() + inverted);
	
	// Adjust the ancestors of the source node.
	ConstNodeIterator sourceParentNode = sourceNode;
	while (!sourceParentNode.contains(destLeaf)) {
		--sourceParentNode.listRef().leafCount;
		sourceParentNode = sourceParentNode.parent();
	}
	
	// Adjust the ancestors of the destination node.
	ConstNodeIterator destParentNode = destNode;
	while (!destParentNode.contains(sourceLeaf)) {
		++destParentNode.listRef().leafCount;
		destParentNode = destParentNode.parent();
	}
	
	// Both sourceParentNode and destParentNode should be equal to the same
	// node: the most recent common ancestor of both the source and
	// destination.
	
	// Adjust the nodes in between the source and destination node.
	std::ptrdiff_t invertedSign = inverted ? -1 : +1;
	for (auto node = sourceNode.listIt(); node < destNode.listIt(); ++node) {
		node->leafIndex -= invertedSign;
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
bool Octree<L, N, Dim>::adjust(
		ConstNodeIterator node) {
	bool result = false;
	// If the node doesn't have children but should, then make them.
	if (!node.hasChildren() && !node.canHoldLeafs(0)) {
		node = createChildren(node);
		result = true;
	}
	// If the node does have children but shouldn't, then remove them.
	else if (node.hasChildren() && node.canHoldLeafs(0)) {
		node = destroyChildren(node);
		result = true;
	}
	// Then, adjust all of this node's children as well.
	if (node.hasChildren()) {
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
			ConstNodeIterator child = node.child(index);
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
	Leaf leaf(data, position);
	auto node = find(hint, leaf.position());
	if (node == nodes().end()) {
		return leafs().end();
	}
	// Create children if the node doesn't have the capacity to store
	// this leaf.
	if (_adjust && !canHoldLeafs(node, +1)) {
		node = createChildren(node);
		node = find(node, leaf.position());
	}
	return std::make_tuple(node, insertAt(node, leaf));
}

template<typename L, typename N, std::size_t Dim>
std::tuple<
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::insert(
		L const& data,
		Vector<Dim> const& position) {
	return insert(nodes().front(), data, position);
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
	auto node = find(hint, leaf);
	if (node == _nodes.end()) {
		return _leafs.end();
	}
	// If the parent of this node doesn't need to be divided into subnodes
	// anymore, then merge its children together.
	while (
			_adjust &&
			node->hasParent &&
			canHoldLeafs(node + node->parentIndex, -1)) {
		node = destroyChildren(node + node->parentIndex);
	}
	return std::make_tuple(node, eraseAt(node, leaf));
}

template<typename L, typename N, std::size_t Dim>
std::tuple<
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::erase(
		LeafIterator leaf) {
	return erase(nodes().root(), leaf);
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
	auto source = find(hint, leaf);
	auto dest = find(hint, position);
	if (source == _nodes.end() || dest == _nodes.end()) {
		return _leafs.end();
	}
	// If the source and the destination are distinct, then check to make
	// sure that they remain within the node capcity.
	if (_adjust && source != dest) {
		while (
				source->hasParent &&
				canHoldLeafs(source + source->parentIndex, -1)) {
			std::size_t destIndex = dest - _nodes.begin();
			if (dest > source) {
				destIndex -= (1 << Dim);
			}
			source = destroyChildren(source + source->parentIndex);
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

template<typename L, typename N, std::size_t Dim>
std::tuple<
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::NodeIterator,
		typename Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::move(
		LeafIterator leaf,
		Vector<Dim> const& position) {
	return move(nodes().root(), leaf, position);
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeIterator Octree<L, N, Dim>::find(
		ConstNodeIterator hint,
		Vector<Dim> const& position) {
	bool contains = hint.contains(position);
	// If this node is childless and contains the point, then just return it.
	if (contains && !hint.hasChildren()) {
		return hint;
	}
	// If it is childless but contains the point, then recursively call this
	// method on the child that also conains the point.
	else if (contains && hint.hasChildren()) {
		return find(hint.child(position), position);
	}
	// If it does not contain the point, then recursively call this method on
	// the parent of this node.
	else if (hint.hasParent()) {
		return find(hint.parent(), position);
	}
	return nodes().end();
}

template<typename L, typename N, std::size_t Dim>
typename Octree<L, N, Dim>::NodeIterator Octree<L, N, Dim>::find(
		ConstNodeIterator hint,
		ConstLeafIterator leaf) {
	bool contains = hint.contains(leaf);
	// If this node is childless and contains the leaf, then return itself.
	if (contains && !hint.hasChildren()) {
		return hint;
	}
	// If it has children and contains the leaf, then recursively call this
	// method on the child that contains the leaf.
	else if (contains && hint.hasChildren()) {
		return find(hint.child(leaf), leaf);
	}
	// If it doesn't contain the leaf, the recursively call this method on the
	// parent of this node.
	else if (hint.hasParent()) {
		return find(hint.parent(), leaf);
	}
	return nodes().end();
}

// *--------------------------*
// | NodeIteratorBase methods |
// *--------------------------*

template<typename L, typename N, std::size_t Dim>
template<bool Const, bool Reverse>
typename Octree<L, N, Dim>::template NodeRangeBase<Const>
Octree<L, N, Dim>::NodeIteratorBase<Const, Reverse>::children() const {
	return NodeRangeBase<Const>(
		_octree,
		_index + listRef().childIndices[0],
		_index + listRef().childIndices[1 << Dim]);
}

template<typename L, typename N, std::size_t Dim>
template<bool Const, bool Reverse>
typename Octree<L, N, Dim>::template NodeRangeBase<Const>
Octree<L, N, Dim>::NodeIteratorBase<Const, Reverse>::nodes() const {
	return NodeRangeBase<Const>(
		_octree,
		_index,
		_index + listRef().childIndices[1 << Dim]);
}

template<typename L, typename N, std::size_t Dim>
template<bool Const, bool Reverse>
typename Octree<L, N, Dim>::template LeafRangeBase<Const>
Octree<L, N, Dim>::NodeIteratorBase<Const, Reverse>::leafs() const {
	return LeafRangeBase<Const>(
		_octree,
		listRef().leafIndex,
		listRef().leafIndex + listRef().leafCount);
}

#endif

