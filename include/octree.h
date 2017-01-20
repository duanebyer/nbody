#ifndef __NBODY_OCTREE_H_
#define __NBODY_OCTREE_H_

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensor.h"

namespace nbody {

/**
 * \brief Wraps a piece of data together with a position so that it can be
 * stored at the leaf of an Octree.
 * 
 * \tparam Data the type of data stored at the leaves of the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename Data, std::size_t Dim>
class OctreeData final {
	template<typename Data, typename Node, std::size_t Num, std::size_t Dim>
	friend class Octree;
	template<typename Data, typename Node, std::size_t Num, std::size_t Dim>
	friend class OctreeNode;
	
private:
	
	Data _data;
	Vector<Dim> _position;
	
	OctreeData(Data data, Vector<Dim> position) :
			_data(data),
			_position(position) {
	}
	
public:
	
	Data& data() {
		return _data;
	}
	Data const& data() const {
		return _data;
	}
	
	Vector<Dim> position() const {
		return _position;
	}
	
};

/**
 * \brief Represents an Octree node containing child nodes and possibly data.
 * 
 * Each OctreeNode may contain a certain number of children (the number of
 * children depends upon the dimension of the space). If the OctreeNode does
 * not have any children, then it is considered a leaf. Each leaf may store a
 * single data point.
 * 
 * In addition, each OctreeNode has node data associated with it, regardless of
 * whether it is a leaf or not. This data could contain, for example, the
 * center of mass of all of the children of the OctreeNode. Every time a
 * change is made to the OctreeNode or its children, its internal data is also
 * updated.
 * 
 * \tparam Data the type of data stored at the leaves of the Octree
 * \tparam Node the type of data stored at the nodes of the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename Data, typename Node, std::size_t Dim>
class OctreeNode final {
	
	friend class Octree<Data, Node, Dim>;
	
private:
	
	// All OctreeNodes are guaranteed to be stored contigiously. OctreeNodes
	// can only be referred to by reference or by pointer. The memory location
	// of an OctreeNode is guaranteed to be within an Octree's list of nodes.
	
	// The octree that contains this node.
	Octree<Data, Node, Num, Dim> const* _octree;
	
	// The depth of this node within the octree (0 for root, and so on).
	std::size_t _depth;
	
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
	
	// The number of data points that this node contains. This includes data
	// points stored by all descendants of this node.
	std::size_t _dataCount;
	// The data points that this node contains are located at this index.
	std::size_t _dataIndex;
	
	// The section of space that this node encompasses.
	Vector<Dim> _position;
	Vector<Dim> _dimensions;
	
	OctreeNode(Octree<Data, Node, Dim> const* octree) :
			_octree(octree),
			_depth(0),
			_hasChildren(false),
			_childIndices(),
			_hasParent(false),
			_dataCount(0),
			_dataIndex(0),
			_position(),
			_dimensions() {
	}
	
	OctreeNode(OctreeNode<Data, Node, Dim> const&) = delete;
	void operator=(OctreeNode<Data, Node, Dim> const&) = delete;
	
	// Returns whether the node can accomodate a change in the number of data
	// points by 'n' and still hold all of the data points without children.
	bool canHoldData(std::ptrdiff_t n = 0) const {
		// If the depth of the node is larger than the max, then it has
		// infinite capacity.
		return
			_dataCount + n < _octree->_nodeCapacity ||
			_depth >= _octree->_maxDepth;
	}
	
	// Divides this node into a set of subnodes and partitions its data between
	// them. This function may reorganize the data vector (some data iterators
	// may become invalid).
	void createChildren() {
		// Create the 2^Dim child nodes inside the parent octree.
		auto begin = _octree->_nodes.begin();
		auto node = begin + (this - &_octree->_nodes.front());
		auto firstChild = _octree->_nodes.insert(
			node, 1 << Dim,
			OctreeNode<Data, Node, Dim>(_octree));
		
		// Update the iterators as they have been invalidated.
		begin = _octree->_nodes.begin();
		node = begin + (this - &_octree->_nodes.front());
		
		// Loop through the new children, and set up their various properties.
		Vector<Dim> childDimensions = _dimensions / 2;
		Vector<Dim> midpoint = _position + childDimensions;
		auto child = firstChild;
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
			child->_depth = _depth + 1;
			child->_hasParent = true;
			child->_parentIndex = -((std::ptrdiff_t) index + 1);
			child->_siblingIndex = index;
			child->_dataIndex = _dataIndex + _dataCount;
			child->_dimensions = childDimensions;
			child->_position = _position;
			// Shift the child position depending on which child it is.
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				if ((1 << dim) & index) {
					child->_position[dim] = midpoint[dim];
				}
			}
			++child;
			
			// Add the child to this node.
			_childIndices[index] = index + 1;
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
		
		// Distribute the data of this node to the children.
		for (std::size_t index = 0; index < _dataCount; ++index) {
			// Figure out which node the data belongs to.
			std::childIndex = 0;
			auto data = _octree->_data.begin() + _dataIndex + index;
			Vector<Dim> const& position = data->position();
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				if (position[dim] >= midpoint[dim]) {
					childIndex += (1 << dim);
				}
			}
			auto owner = find(data);
			owner->moveAt(data, firstChild + childIndex);
		}
	}
	
	// Destroys all descendants of this node and takes their data into this
	// node. This function will not reorganize the data vector (all data
	// iterators will remain valid).
	void destroyChildren() {
		// Determine how many children, grandchildren, great-grandchildren, ...
		// of this node.
		std::size_t numDescendants = _childIndices[1 << Dim];
		auto begin = _octree->_nodes.begin();
		auto node = begin + (this - &_octree->_nodes.front());
		
		// Destroy the subnodes and update iterators.
		_octree->_nodes->erase(node + 1, node + 1 + numDescendants);
		begin = _octree->_nodes.begin();
		node = begin + (this - &_octree->nodes.front());
		
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
	
	// Adds a piece of data to this specific node.
	Octree<Node, Data, Dim>::DataIterator insertAt(
			OctreeData<Data, Dim> const& data) {
		// Add the data to the master list of data in the octree and update
		// internal variables.
		auto begin = _octree->_data.begin();
		auto newData = _octree->_data.insert(begin + _dataIndex + _dataCount);
		
		// Loop through the rest of the nodes and increment their data indices
		// so that they still refer to the correct location in the data vector.
		auto begin = _octree->_nodes.begin();
		auto end = _octree->_nodes.end();
		auto node = begin + (this - &_octree->nodes.first());
		auto currentNode = node;
		while (++currentNode != end) {
			++currentNode->_dataIndex;
		}
		
		// Also loop through all ancestors and increment their data counts.
		auto parent = node;
		while (parent->_hasParent) {
			++parent->_dataCount;
			parent += parent->_parentIndex;
		}
		
		return newData;
	}
	
	// Removes a piece of data from this node.
	Octree<Node, Data, Dim>::DataIterator eraseAt(
			Octree<Node, Data, Dim>::ConstDataIterator data) {
		// Remove the data from the master octree data vector.
		auto next = _octree->_data.erase(data);
		
		// Loop through the rest of the nodes and increment their data indices
		// so that they still refer to the correct location in the data vector.
		auto begin = _octree->_nodes.begin();
		auto end = _octree->_nodes.end();
		auto node = begin + (this - &_octree->nodes.first());
		auto currentNode = node;
		while (++currentNode != end) {
			--currentNode->_dataIndex;
		}
		
		// Loop through all of the ancestors of this node and decremement their
		// data counts.
		auto parent = node;
		while (parent->_hasParent) {
			--parent->_dataCount;
			parent += parent->_parentIndex;
		}
		
		return next;
	}
	
	// Moves a piece of data from this node to another one.
	Octree<Node, Data, Dim>::DataIterator moveAt(
			Octree<Node, Data, Dim>::ConstDataIterator sourceData,
			Octree<Node, Data, Dim>::ConstNodeIterator destNode) {
		// Reinsert the data into the data vector in its new position.
		auto beginData = _octree->_data.begin();
		auto destData = beginData + destNode->_dataIndex + destNode->_dataCount;
		bool inverted = sourceData > destData;
		auto firstData = inverted ? destData : sourceData;
		auto lastData = inverted ? sourceData : destData;
		
		std::size_t sourceDataIndex = sourceData - beginData;
		std::size_t destDataIndex = destData - beginData;
		
		std::rotate(firstData, sourceData + !inverted, lastData + inverted);
		
		// Adjust the ancestors of the source node.
		auto beginNode = _octree->_nodes.begin();
		auto sourceNode = beginNode + (this - &_nodes.first());
		auto sourceParentNode = sourceNode;
		std::size_t sourceChildIndex = 0;
		while (!(
				destDataIndex >= sourceParentNode->_dataIndex &&
				destDataIndex < sourceParentNode->_dataIndex +
				                sourceParentNode->_dataCount)) {
			--sourceParentNode->_dataCount;
			std::size_t siblingIndex = sourceParentNode->_siblingIndex;
			sourceParentNode += sourceParentNode->_parentIndex;
			sourceChildIndex += sourceParentNode->_childIndices[siblingIndex];
		}
		
		// Adjust the ancestors of the destination node.
		auto destParentNode = destNode;
		std::size_t destChildIndex = 0;
		while (!(
				sourceDataIndex >= destParentNode->_dataIndex &&
				sourceDataIndex < destParentNode->_dataIndex +
				                  destParentNode->_dataCount)) {
			++destParentNode->_dataCount;
			std::size_t siblingIndex = destParentNode->_siblingIndex;
			destParentNode += destParentNode->_parentIndex;
			destChildIndex += destParentNode->_childIndices[siblingIndex];
		}
		
		// Both sourceParentNode and destParentNode should be equal to the same
		// node: the most recent common ancestor of both the source and
		// destination.
		
		// Adjust the nodes in between the source and destination node.
		for (
				std::size_t nodeIndex = sourceChildIndex;
				nodeIndex < destChildIndex;
				nodeIndex += 2 * !inverted - 1) {
			auto node = sourceParentNode + nodeIndex;
			node->_dataIndex += 2 * inverted - 1;
		}
		
		return beginData + destDataIndex;
	}
	
public:
	
	bool adjust() {
		if (!_hasChildren && !canHoldData(0)) {
			createChildren();
		}
		else if (_hasChildren && canHoldData(0)) {
			destroyChildren();
		}
		else if (_hasChildren) {
			auto begin = _octree->_nodes.begin();
			auto node = begin + (this - &_octree->_nodes.first());
			for (std::size_t index = 0; index < (1 << Dim); ++index) {
				auto child = node + _childIndices[index];
				child->adjust();
			}
		}
	}
	
	Octree<Data, Node, Dim>::DataIterator insert(
			OctreeData<Data, Dim> const& data) {
		// Find the node with the correct position, and insert the data into
		// that node.
		auto node = find(data.position());
		if (node == _octree->_nodes.end()) {
			return _octree->_data.end();
		}
		// Create children if the node doesn't have the capacity to store
		// this data point.
		if (_octree->_adjust && !canHoldData(+1)) {
			node->createChildren();
			node = node->find(data.position());
		}
		return node->insertAt(data);
	}
	
	Octree<Data, Node, Dim>::DataIterator insert(
			Data const& data,
			Vector<Dim> const& position) {
		OctreeData<Data, Dim> octreeData(data, position);
		return insert(octreeData);
	}
	
	Octree<Data, Node, Dim>::DataIterator erase(
			Octree<Data, Node, Dim>::ConstDataIterator it) {
		// Find the node that contains this data, and then erase the data from
		// that node.
		auto node = find(it);
		if (node == _octree->_nodes.end()) {
			return _octree->_data.end();
		}
		if (_octree->_adjust && node->_hasParent) {
			auto parent = node + node->_parentIndex;
			if (parent->canHoldData(-1)) {
				parent->destroyChildren();
				node = parent;
			}
		}
		return node->eraseAt(data);
	}
	
	Octree<Data, Node, Dim>::DataIterator erase(
			Octree<Data, Node, Dim>::DataIterator it) {
		Octree<Data, Node, Dim>::ConstDataIterator constIt = it;
		return eraseData(constIt);
	}
	
	Octree<Data, Node, Dim>::DataIterator move(
			Octree<Data, Node, Dim>::DataIterator it,
			Vector<Dim> position) {
		// Find the source node that contains the data, and the target node
		// with the correct position.
		auto source = find(it);
		auto dest = find(position);
		if (source == _octree->_nodes.end() || dest == _octree->_nodes.end()) {
			return _octree->_data.end();
		}
		// If the source and the destination are distinct, then check to make
		// sure that they remain within the node capcity.
		if (_octree->_adjust && source != dest) {
			if (source->_hasParent) {
				auto parent = source + source->_parentIndex;
				if (parent->canHoldData(-1)) {
					parent->destroyChildren();
					source = parent;
				}
			}
			if (!dest->canHoldData(+1)) {
				dest->createChildren();
				dest = dest->find(position);
			}
		}
		return source->moveAt(dest, it);
	}
	
	Octree<Data, Node, Dim>::NodeIterator find(
			Vector<Dim> position) {
		bool contains = contains(position);
		auto begin = _octree->_nodes.begin();
		auto node = begin + (this - &_octree->_nodes.first());
		if (contains && !_hasChildren) {
			return node;
		}
		else if (contains && _hasChildren) {
			auto child = node + 1;
			Vector<Dim> midpoint = _position + _dimensions / 2;
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				if (position[dim] >= midpoint[dim]) {
					child += (1 << dim);
				}
			}
			return child->find(position);
		}
		else if (_hasParent) {
			auto parent = node + _parentIndex;
			return parent->find(position);
		}
		return _octree->_nodes.end();
	}
	
	Octree<Data, Node, Dim>::ConstNodeIterator find(
			Vector<Dim> position) const {
		return const_cast<OctreeNode<Data, Node, Dim>*>(this)->find(position);
	}
	
	Octree<Data, Node, Dim>::NodeIterator find(
			Octree<Data, Node, Dim>::ConstDataIterator it) {
		std::size_t dataIndex = it - _octree->_data.begin();
		std::size_t lower = _dataIndex;
		std::size_t upper = lower + _dataCount;
		
		bool contains = dataIndex >= lower && dataIndex < upper;
		
		auto begin = _octree->_nodes.begin();
		auto node = begin + (this - &_octree->_nodes.first());
		
		if (contains && !_hasChildren) {
			return node;
		}
		else if (contains && _hasChildren) {
			for (std::size_t index = 0; index < (1 << Dim); ++index) {
				auto child = node + _childIndices[index];
				std::size_t lower = child->_dataIndex;
				std::size_t upper = lower + child->_dataCount;
				if (dataIndex >= lower && dataIndex < upper) {
					return child->find(data);
				}
			}
		}
		else if (_hasParent) {
			auto parent = node + _parentIndex;
			return parent->find(it);
		}
		return _octree->_nodes.end();
	}
	
	Octree<Data, Node, Dim>::NodeIterator find(
			Octree<Data, Node, Dim>::DataIterator it) {
		Octree<Data, Node, Dim>::ConstNodeIterator constIt = it;
		return find(constIt);
	}
	
	Octree<Data, Node, Dim>::ConstNodeIterator find(
			Octree<Data, Node, Dim>::ConstDataIterator it) const {
		return const_cast<OctreeNode<Data, Node, Dim>*>(this)->find(it);
	}
	
	Octree<Data, Node, Dim>::ConstNodeIterator find(
			Octree<Data, Node, Dim>::DataIterator it) const {
		return const_cast<OctreeNode<Data, Node, Dim>*>(this)->find(it);
	}
	
	bool contains(Vector<Dim> const& point) {
		Vector<Dim> lower = _position;
		Vector<Dim> upper = _position + _dimensions;
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			// De Morgan's law has been applied here so that NaN is dealt with
			// appropriately.
			if (!(point[dim] >= lower[dim] && point[dim] < upper[dim])) {
				return false;
			}
		}
		return true;
	}
	
	bool hasChildren() const {
		return _hasChildren;
	}
	
	bool hasParent() const {
		return _hasParent;
	}
	
	Octree<Data, Node, Dim>::NodeIterator iterator() {
		auto begin = _octree->_nodes.begin();
		auto node = begin + (this - &_octree->_nodes.first());
		return node;
	}
	
	Octree<Data, Node, Dim>::ConstNodeIterator iterator() const {
		auto begin = _octree->_nodes.begin();
		auto node = begin + (this - &_octree->_nodes.first());
		return node;
	}
	
	Octree<Data, Node, Dim>::NodeIterator parent() {
		auto node = iterator();
		if (!node->_hasParent) {
			return _octree->_nodes.end();
		}
		else {
			return node + node->_parentIndex;
		}
	}
	Octree<Data, Node, Dim>::ConstNodeIterator parent() const {
		auto node = iterator();
		if (!node->_hasParent) {
			return _octree->_nodes.end();
		}
		else {
			return node + node->_parentIndex;
		}
	}
	
	OctreeNode<T, Dim>* children() {
		return _children;
	}
	OctreeNode<T, Dim> const* children() const {
		return _children;
	}
	
	OctreeData<T, Dim>& data() {
		return _octree->_data[_dataIndex];
	}
	OctreeData<T, Dim> const& data() const {
		return _octree->_data[_dataIndex];
	}
	
	Vector<Dim> position() const {
		return _position;
	}
	
	Vector<Dim> dimensions() const {
		return _dimensions;
	}
	
};

/**
 * \brief A data structure that stores positional data in arbitrary
 * dimensional space.
 * 
 * \tparam Data the type of data stored at the leaves of the Octree
 * \tparam Node the type of data stored at the nodes of the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename Data, typename Node, std::size_t Dim>
class Octree final {
	
private:
	
	// A list storing all of the nodes of the octree.
	std::vector<OctreeNode<Dim, Node, Dim> > _nodes;
	
	// A list storing all of the actual data.
	std::vector<OctreeData<Data, Dim> > _data
	
	// The number of data points to store at a single node of the octree.
	std::size_t _nodeCapacity;
	
	// The maximum depth of the octree.
	std::size_t _maxDepth;
	
	// Whether the tree should be automatically readjust itself so that each
	// node has less data than the node capacity. If this is false, then the
	// adjust() method has to be called to force an adjustment.
	bool _adjust;
	
public:
	
	using NodeIterator = decltype(_nodes)::iterator;
	using ConstNodeIterator = decltype(_nodes)::const_iterator;
	using DataIterator = decltype(_data)::iterator;
	using ConstDataIterator = decltype(_data)::const_iterator;
	
	Octree(Vector<Dim> position, Vector<Dim> dimensions) :
			_nodes(),
			_data(),
			_nodeCapacity(1) {
	}
	
	/**
	 * \brief Returns the root OctreeNode of the entire Octree.
	 */
	OctreeNode<T, N, Dim>& root() {
		return _nodes[0];
	}
	OctreeNode<T, N, Dim> const& root() const {
		return _nodes[0];
	}
	
	/**
	 * \brief Adds a new piece of data to the Octree.
	 * 
	 * \param data the data to be added
	 * \param position the position at which the data should be added
	 * \return an iterator to the position at which the data is stored
	 */
	Iterator insert(T data, Vector<Dim> position) {
		std::size_t dataIndex = _nodes.size();
		_data.push_back(OctreeData<T, N, Dim>(NULL, data, position));
		_root.insert(&_data.back());
		return _data.end() - 1;
	}
	
	/**
	 * \brief Removes a piece of data from the Octree.
	 * 
	 * \param it an iterator to the data that should be removed
	 * \return an iterator to the data following the removed data
	 */
	Iterator erase(ConstIterator it) {
		// Remove the data from the vector, and then loop through and decrease
		// the data index for the parent of every following piece of data.
		it->_parent->erase();
		Iterator eraseIt = _data->erase(it);
		Iterator end = _data.end();
		for (Iterator loopIt = eraseIt; loopIt != end; ++loopIt) {
			loopIt->_parent->_dataIndex -= 1;
		}
		return eraseIt;
	}
	
	Iterator erase(Iterator it) {
		return erase((ConstIterator) it);
	}
	
	/**
	 * \brief Moves an existing piece of data from one position in the Octree
	 * to another.
	 * 
	 * \param it an iterator to the data that should be moved
	 * \param position the new position of the data
	 * \return an iterator to the new data
	 */
	Iterator move(ConstIterator it, Vector<Dim> position) {
		OctreeNode<T, N, Dim>* parent = it->_parent;
		std::size_t dataIndex = parent->_dataIndex;
		it->_position = position;
		// If the parent no longer contains the data, then remove the data from
		// its current parent, and go up the tree until a node is reached that
		// can store the data.
		if (!parent->contains(it->_position)) {
			parent->erase();
			OctreeNode<T, N, Dim>* node = parent;
			do {
				node = node->_parent;
			} while (!node->contains(it->_position));
			node->insert(&(*it));
		}
		return it;
	}
	
	Iterator move(Iterator it, Vector<Dim> position) {
		return move((ConstIterator) it, position);
	}
	
	Iterator begin() {
		return _data.begin();
	}
	ConstIterator begin() const {
		return _data.begin();
	}
	
	Iterator end() {
		return _data.end();
	}
	ConstIterator end() const {
		return _data.end();
	}
	
	OctreeData<T, N, Dim>* data() {
		return _data;
	}
	
	OctreeData<T, N, Dim> const* data() const {
		return _data;
	}
	
};

}

#endif

