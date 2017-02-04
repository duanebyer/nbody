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
	leafs.insert(leafs.begin(), root);
}

template<
		typename L, typename N, std::size_t Dim,
		IterationOrder Order, bool Const, bool Reverse>
bool Octree<L, N, Dim>::canHoldLeafs(
		Octree<L, N, Dim>::NodeIteratorBase<Order, Const, Reverse> nodeIt,
		std::ptrdiff_t n) const {
	// If the depth of the node is larger than the max, then it has infinite
	// capacity.
	auto node = nodeIt.listIt();
	return
		node->dataCount + n < _nodeCapacity ||
		node->depth >= _maxDepth;
}

template<
		typename L, typename N, typename Dim,
		IterationOrder Order, bool Const, bool Reverse>
Octree<L, N, Dim>::NodeIteratorBase<Order, false, Reverse>
Octree<L, N, Dim>::createChildren(
		Octree<L, N, Dim>NodeIteratorBase<Order, Const, Reverse> nodeIt) {
	auto node = nodeIt.listIt();
	// Create the 2^Dim child nodes inside the parent octree.
	auto firstChild = _nodes.insert(node, 1 << Dim, Node());
	
	// Update the node iterator as it has been invalidated.
	node = firstChild - 1;
	
	// Loop through the new children, and set up their various properties.
	Vector<Dim> childDimensions = node->dimensions / 2;
	Vector<Dim> midpoint = node->position + childDimensions;
	auto child = firstChild;
	for (std::size_t index = 0; index < (1 << Dim); ++index) {
		child->depth = node->depth + 1;
		child->hasParent = true;
		child->parentIndex = -((std::ptrdiff_t) index + 1);
		child->siblingIndex = index;
		child->leafIndex = node->leafIndex + node->leafCount;
		child->dimensions = childDimensions;
		child->position = node->position;
		// Shift the child position depending on which child it is.
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if ((1 << dim) & index) {
				child->position[dim] = midpoint[dim];
			}
		}
		++child;
		
		// Add the child to this node.
		node->childIndices[index] = index + 1;
	}
	
	// Go through the parent, grandparent, great-grandparent, ...  of this
	// node and update their child indices.
	auto parent = node;
	while (parent->hasParent) {
		std::size_t siblingIndex = parent->siblingIndex;
		parent += parent->parentIndex;
		while (++siblingIndex <= (1 << Dim)) {
			parent->childIndices[siblingIndex] += (1 << Dim);
			auto child = parent + parent->childIndices[siblingIndex];
			child->parentIndex -= (1 << Dim);
		}
	}
	
	// Distribute the leaves of this node to the children.
	for (std::size_t index = 0; index < node->leafCount; ++index) {
		// Figure out which node the leaf belongs to.
		std::size_t childIndex = 0;
		auto leaf = _leafs.begin() + node->leafIndex + index;
		Vector<Dim> const& position = leaf->position();
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (position[dim] >= midpoint[dim]) {
				childIndex += (1 << dim);
			}
		}
		moveAt(node, firstChild + childIndex, leaf);
	}
	
	return NodeIteratorBase<Order, Const, Reverse>(node);
}

template<
		typename L, typename N, std::size_t Dim,
		IterationOrder Order, bool Const, bool Reverse>
Octree<L, N, Dim>::NodeIteratorBase<Order, Const, Reverse>
Octree<L, N, Dim>::destroyChildren(
		Octree<L, N, Dim>::NodeIteratorBase<Order, Const, Reverse> nodeIt) {
	auto node = nodeIt.listIt();
	// Determine how many children, grandchildren, great-grandchildren, ...
	// of this node.
	std::size_t numDescendants = node->childIndices[1 << Dim];
	
	// Destroy the subnodes and update iterators.
	auto nextNode = _nodes->erase(node + 1, node + 1 + numDescendants);
	node = nextNode - 1;
	
	// Go through the parent, grandparent, great-grandparent, ... of this
	// node and update their child indices.
	auto parent = node;
	while (parent->hasParent) {
		std::size_t siblingIndex = parent->siblingIndex;
		parent += parent->parentIndex;
		while (++siblingIndex <= (1 << Dim)) {
			parent->childIndices[siblingIndex] -= numDescendants;
			auto child = parent + parent->childIndices[siblingIndex];
			child->parentIndex += numDescendants;
		}
	}
	
	return NodeIteratorBase<Order, Const, Reverse>(node);
}

template<
		typename L, typename N, std::size_t Dim,
		IterationOrder NOrder, bool NConst, bool NReverse,
		bool LReverse>
Octree<L, N, Dim>::LeafIteratorBase<false, LReverse>
Octree<L, N, Dim>::insertAt(
		Octree<L, N, Dim>::NodeIteratorBase<NOrder, NConst, NReverse> nodeIt,
		Leaf const& leaf) {
	auto node = nodeIt.listIt();
	// Add the leaf to the master list of leaves in the octree and update
	// internal variables.
	auto newLeaf = _leafs.insert(
		_leafs.begin + node->leafIndex + node->leafCount);
	
	// Loop through the rest of the nodes and increment their leaf indices
	// so that they still refer to the correct location in the leaf vector.
	auto currentNode = node;
	while (++currentNode != _nodes.end()) {
		++currentNode->leafIndex;
	}
	
	// Also loop through all ancestors and increment their leaf counts.
	auto parent = node;
	while (parent->hasParent) {
		++parent->leafCount;
		parent += parent->parentIndex;
	}
	
	return LeafIteratorBase<false, LReverse>(newLeaf);
}

template<
		typename L, typename N, std::size_t Dim,
		IterationOrder NOrder, bool NConst, bool NReverse,
		bool LConst, bool LReverse>
Octree<L, N, Dim>::LeafIteratorBase<false, LReverse>
Octree<L, N, Dim>::eraseAt(
		Octree<L, N, Dim>::NodeIteratorBase<NOrder, NConst, NReverse> nodeIt,
		Octree<L, N, Dim>::LeafIteratorBase<LConst, LReverse> leafIt) {
	auto node = nodeIt.listIt();
	auto leaf = leafIt.listIt();
	// Remove the leaf from the master octree leaf vector.
	auto next = _leafs.erase(leaf);
	
	// Loop through the rest of the nodes and increment their leaf indices
	// so that they still refer to the correct location in the leaf vector.
	auto currentNode = node;
	while (++currentNode != _nodes.end()) {
		--currentNode->leafIndex;
	}
	
	// Loop through all of the ancestors of this node and decremement their
	// leaf counts.
	auto parent = node;
	while (parent->hasParent) {
		--parent->leafCount;
		parent += parent->parentIndex;
	}
	
	return LeafIteratorBase<false, LReverse>(next);
}

template<
		typename L, typename N, std::size_t Dim,
		IterationOrder N1Order, bool N1Const, bool N1Reverse,
		IterationOrder N2Order, bool N2Const, bool N2Reverse,
		bool LConst, bool LReverse>
Octree<L, N, Dim>::LeafIterator<false, LReverse> Octree<L, N, Dim>::moveAt(
		Octree<L, N, Dim>::NodeIteratorBase<N1Order, N1Const, N1Reverse>
		sourceNodeIt,
		Octree<L, N, Dim>::NodeIteratorBase<N2Order, N2Const, N2Reverse>
		destNodeIt,
		Octree<L, N, Dim>::LeafIteratorBase<LConst, LReverse>
		sourceLeafIt) {
	auto sourceNode = sourceNodeIt.listIt();
	auto destNode = destNodeIt.listIt();
	auto sourceLeaf = sourceLeafIt.listIt();
	// Reinsert the leaf into the leaf vector in its new position.
	auto destLeaf =
		_leafs.begin() + destNode->leafIndex + destNode->leafCount;
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
			destLeafIndex >= sourceParentNode->leafIndex &&
			destLeafIndex < sourceParentNode->leafIndex +
					sourceParentNode->leafCount)) {
		--sourceParentNode->leafCount;
		std::size_t siblingIndex = sourceParentNode->siblingIndex;
		sourceParentNode += sourceParentNode->parentIndex;
		sourceChildIndex += sourceParentNode->childIndices[siblingIndex];
	}
	
	// Adjust the ancestors of the destination node.
	auto destParentNode = destNode;
	std::size_t destChildIndex = 0;
	while (!(
			sourceLeafIndex >= destParentNode->leafIndex &&
			sourceLeafIndex < destParentNode->leafIndex +
					  destParentNode->leafCount)) {
		++destParentNode->leafCount;
		std::size_t siblingIndex = destParentNode->siblingIndex;
		destParentNode += destParentNode->parentIndex;
		destChildIndex += destParentNode->childIndices[siblingIndex];
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
		node->leafIndex -= invertedSign;
	}
	
	return LeafIteratorBase<LConst, LReverse>(_leafs.begin() + destLeafIndex);
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
bool Octree<L, N, Dim>::adjust(
		Octree<L, N, Dim>::ConstNodeIterator<Order> node) {
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
		auto begin = _nodes.begin();
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
			auto child = node + node->childIndices[index];
			bool nextResult = adjust(child);
			result = result || nextResult;
		}
	}
	return result;
}

template<typename L, typename N, std::size_t Dim>
bool Octree<L, N, Dim>::adjust() {
	return adjust(root());
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
std::tuple<
		Octree<L, N, Dim>::NodeIterator<Order>,
		Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::insert(
		Octree<L, N, Dim>::ConstNodeIterator<Order> start,
		L const& data,
		Vector<Dim> const& position) {
	// Find the node with the correct position, and insert the leaf into
	// that node.
	Leaf leaf(data, position);
	auto node = find(start, leaf.position());
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

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
std::tuple<
		Octree<L, N, Dim>::NodeIterator<Order>,
		Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::insert(
		L const& data,
		Vector<Dim> const& position) {
	return insert(root(), data, position);
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
std::tuple<
		Octree<L, N, Dim>::NodeIterator<Order>,
		Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::erase(
		Octree<L, N, Dim>::ConstNodeIterator<Order> start,
		Octree<L, N, Dim>::LeafIterator leaf) {
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
			node->hasParent &&
			canHoldLeafs(node + node->parentIndex, -1)) {
		node = destroyChildren(node + node->parentIndex);
	}
	return std::make_tuple(node, eraseAt(node, leaf));
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
std::tuple<
		Octree<L, N, Dim>::NodeIterator<Order>,
		Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::erase(
		Octree<L, N, Dim>::LeafIterator leaf) {
	return erase(root(), leaf);
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
std::tuple<
		Octree<L, N, Dim>::NodeIterator<Order>,
		Octree<L, N, Dim>::NodeIterator<Order>,
		Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::move(
		Octree<L, N, Dim>::ConstNodeIterator<Order> start,
		Octree<L, N, Dim>::LeafIterator leaf,
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

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
std::tuple<
		Octree<L, N, Dim>::NodeIterator<Order>,
		Octree<L, N, Dim>::NodeIterator<Order>,
		Octree<L, N, Dim>::LeafIterator>
Octree<L, N, Dim>::move(
		Octree<L, N, Dim>::LeafIterator leaf,
		Vector<Dim> const& position) {
	return move(root(), leaf, position);
}

template<typename L, typename N, std::size_t Dim>
bool contains(ConstNodeIterator node, Vector<Dim> const& point) {
	Vector<Dim> lower = node->position;
	Vector<Dim> upper = node->position + node->dimensions;
	for (std::size_t dim = 0; dim < Dim; ++dim) {
		// De Morgan's law has been applied here so that NaN is dealt with
		// appropriately.
		if (!(point[dim] >= lower[dim] && point[dim] < upper[dim])) {
			return false;
		}
	}
	return true;
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
Octree<L, N, Dim>::NodeIterator<Order> Octree<L, N, Dim>::find(
		Octree<L, N, Dim>::ConstNodeIterator<Order> startIt,
		Vector<Dim> const& position) {
	auto start = startIt.listIt();
	
	bool contains = start.contains(position);
	auto node = start;
	// If this node is childless and contains the point, then just return it.
	if (contains && !node->hasChildren) {
		return NodeIterator<Order>(node);
	}
	// If it is childless but contains the point, then recursively call this
	// method on the child that also conains the point.
	else if (contains && node->hasChildren) {
		auto child = node + 1;
		Vector<Dim> midpoint = node->position + node->dimensions / 2;
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (position[dim] >= midpoint[dim]) {
				child += (1 << dim);
			}
		}
		return find(ConstNodeIterator<Order>(child), position);
	}
	// If it does not contain the point, then recursively call this method on
	// the parent of this node.
	else if (node->hasParent) {
		auto parent = node + node->parentIndex;
		return find(ConstNodeIterator<Order>(parent), position);
	}
	return nodes().end();
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
Octree<L, N, Dim>::NodeIterator<Order> Octree<L, N, Dim>::find(
		Vector<Dim> const& position) {
	return find(nodes().begin(), position);
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
Octree<L, N, Dim>::ConstNodeIterator<Order> Octree<L, N, Dim>::find(
		Octree<L, N, Dim>::ConstNodeIterator<Order> start,
		Vector<Dim> const& position) const {
	return const_cast<Octree<L, N, Dim>*>(this)->find(
		start, position);
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
Octree<L, N, Dim>::ConstNodeIterator<Order> Octree<L, N, Dim>::find(
		Vector<Dim> const& position) const {
	return find(nodes().begin(), position);
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
Octree<L, N, Dim>::NodeIterator<Order> Octree<L, N, Dim>::find(
		Octree<L, N, Dim>::ConstNodeIterator<Order> startIt,
		Octree<L, N, Dim>::ConstLeafIterator leafIt) {
	auto start = startIt.listIt();
	auto leaf = leafIt.listIt();
	
	auto node = start;
	
	std::size_t leafIndex = leaf - _leafs.begin();
	std::size_t lower = node->leafIndex;
	std::size_t upper = lower + node->leafCount;
	
	bool contains = leafIndex >= lower && leafIndex < upper;
	
	// If this node is childless and contains the leaf, then return itself.
	if (contains && !node->hasChildren) {
		return NodeIterator<Order>(node);
	}
	// If it has children and contains the leaf, then recursively call this
	// method on the child that contains the leaf.
	else if (contains && node->hasChildren) {
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
			auto child = node + node->childIndices[index];
			std::size_t lower = child->leafIndex;
			std::size_t upper = lower + child->leafCount;
			if (leafIndex >= lower && leafIndex < upper) {
				return find(NodeIterator<Order>(child), leafIt);
			}
		}
	}
	// If it doesn't contain the leaf, the recursively call this method on the
	// parent of this node.
	else if (node->hasParent) {
		auto parent = node + node->parentIndex;
		return find(NodeIterator<Order>(parent), leafIt);
	}
	return nodes().end();
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
Octree<L, N, Dim>::NodeIterator<Order> Octree<L, N, Dim>::find(
		Octree<L, N, Dim>::ConstLeafIterator leaf) {
	return find(nodes().begin(), leaf);
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
Octree<L, N, Dim>::ConstNodeIterator<Order> Octree<L, N, Dim>::find(
		Octree<L, N, Dim>::ConstNodeIterator<Order> start,
		Octree<L, N, Dim>::ConstLeafIterator leaf) const {
	return const_cast<Octree<L, N, Dim>*>(this)->find(
		start, leaf);
}

template<typename L, typename N, std::size_t Dim, IterationOrder Order>
Octree<L, N, Dim>::ConstNodeIterator<Order> Octree<L, N, Dim>::find(
		Octree<L, N, Dim>::ConstLeafIterator leaf) const {
	return find(nodes().begin(), leaf);
}

#endif

