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
 * not have any children, then it is considered a leaf. Each leaf may store
 * several data points.
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
	
	OctreeNode() :
			_hasChildren(false),
			_childIndices(),
			_hasParent(false),
			_dataCount(0),
			_dataIndex(0),
			_position(),
			_dimensions() {
	}
	
public:
	
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
	

	// Divides a node into a set of subnodes and partitions its data between
	// them.
	void createChildren(ConstNodeIterator node) {
		// Create the 2^Dim child nodes inside the parent octree.
		auto firstChild = _nodes.insert(
			node, 1 << Dim,
			OctreeNode<Data, Node, Dim>());
		
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
			child->_dataIndex = node->_dataIndex + node->_dataCount;
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
		
		// Distribute the data of this node to the children.
		for (std::size_t index = 0; index < _dataCount; ++index) {
			// Figure out which node the data belongs to.
			std::childIndex = 0;
			auto data = _data.begin() + node->_dataIndex + index;
			Vector<Dim> const& position = data->position();
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				if (position[dim] >= midpoint[dim]) {
					childIndex += (1 << dim);
				}
			}
			moveAt(node, firstChild + childIndex, data);
		}
	}
	
	// Destroys all descendants of a node and takes their data into the node.
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
	
	// Adds a piece of data to a specific node.
	DataIterator insertAt(
			ConstNodeIterator node,
			OctreeData<Data, Dim> const& data) {
		// Add the data to the master list of data in the octree and update
		// internal variables.
		auto newData = _data.insert(
			_data.begin + node->_dataIndex + node->_dataCount);
		
		// Loop through the rest of the nodes and increment their data indices
		// so that they still refer to the correct location in the data vector.
		auto currentNode = node;
		while (++currentNode != _nodes.end()) {
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
	
	// Removes a piece of data from a node.
	DataIterator eraseAt(
			ConstNodeIterator node,
			ConstDataIterator data) {
		// Remove the data from the master octree data vector.
		auto next = _data.erase(data);
		
		// Loop through the rest of the nodes and increment their data indices
		// so that they still refer to the correct location in the data vector.
		auto currentNode = node;
		while (++currentNode != _nodes.end()) {
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
	DataIterator moveAt(
			ConstNodeIterator sourceNode,
			ConstNodeIterator destNode,
			ConstDataIterator sourceData) {
		// Reinsert the data into the data vector in its new position.
		auto destData =
			_data.begin() + destNode->_dataIndex + destNode->_dataCount;
		bool inverted = sourceData > destData;
		auto firstData = inverted ? destData : sourceData;
		auto lastData = inverted ? sourceData : destData;
		
		std::size_t sourceDataIndex = sourceData - beginData;
		std::size_t destDataIndex = destData - beginData;
		
		std::rotate(firstData, sourceData + !inverted, lastData + inverted);
		
		// Adjust the ancestors of the source node.
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
		std::ptrdiff_t invertedSign = inverted ? -1 : +1;
		for (
				std::size_t nodeIndex = sourceChildIndex;
				nodeIndex < destChildIndex;
				nodeIndex += invertedSign) {
			auto node = sourceParentNode + nodeIndex;
			node->_dataIndex -= invertedSign;
		}
		
		return beginData + destDataIndex;
	}
	
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
		return const_cast<Octree<Data, Node, Dim>*>(this)->parent(node);
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
		return const_cast<Octree<Data, Node, Dim>*>(this)->child(node, index);
	}
	
	DataIterator dataBegin() {
		return _data.begin();
	}
	ConstDataIterator dataBegin() const {
		return _data.begin();
	}
	DataIterator dataEnd() {
		return _data.end();
	}
	ConstDataIterator dataEnd() const {
		return _data.end();
	}
	
	DataIterator dataBegin(ConstNodeIterator node) {
		return _data.begin() + node->_dataIndex;
	}
	ConstDataIterator dataBegin(ConstNodeIterator node) const {
		return _data.begin() + node->_dataIndex;
	}
	DataIterator dataEnd(ConstNodeIterator node) {
		return _data.begin() + node->_dataIndex + node->_dataCount;
	}
	ConstDataIterator dataEnd(ConstNodeIterator node) const {
		return _data.begin() + node->_dataIndex + node->_dataCount;
	}
	
	
	
	DataIterator insert(
			ConstNodeIterator start,
			OctreeData<Data, Dim> const& data) {
		// Find the node with the correct position, and insert the data into
		// that node.
		auto node = find(start, data.position());
		if (node == _nodes.end()) {
			return _data.end();
		}
		// Create children if the node doesn't have the capacity to store
		// this data point.
		if (node->_dataCount + 1 >= _nodeCapacity) {
			createChildren(node);
			node = find(node, data.position());
		}
		return insertAt(node, data);
	}
	
	DataIterator insert(
			OctreeData<Data, Dim> const& data) {
		return insert(root(), data);
	}
	
	DataIterator insert(
			ConstNodeIterator start,
			Data const& data,
			Vector<Dim> const& position) {
		OctreeData<Data, Dim> octreeData(data, position);
		return insert(start, octreeData);
	}
	
	DataIterator insert(
			Data const& data,
			Vector<Dim> const& position) {
		return insert(root(), data, position);
	}
	
	
	
	DataIterator erase(
			ConstNodeIterator start,
			DataIterator data) {
		// Find the node that contains this data, and then erase the data from
		// that node.
		auto node = find(start, data);
		if (node == _nodes.end()) {
			return _data.end();
		}
		return eraseAt(node, data);
	}
	
	DataIterator erase(
			DataIterator data) {
		return erase(root(), data);
	}
	
	
	
	DataIterator move(
			ConstNodeIterator start,
			DataIterator data,
			Vector<Dim> const& position) {
		// Find the source node that contains the data, and the target node
		// with the correct position.
		auto source = find(start, data);
		auto dest = find(start, position);
		if (source == _nodes.end() || dest == _nodes.end()) {
			return _data.end();
		}
		// If the source and the destination are distinct, then check to make
		// sure that they remain within the node capcity.
		if (source != dest) {
			if (source->_hasParent) {
				auto parent = source + source->_parentIndex;
				if (parent->_dataCount - 1 < _nodeCapacity) {
					destroyChildren(parent);
					source = parent;
				}
			}
			if (dest->_dataCount + 1 >= _octree->_nodeCapacity) {
				createChildren(dest);
				dest = find(dest, position);
			}
		}
		return moveAt(source, dest, data);
	}
	
	DataIterator move(
			DataIterator data,
			Vector<Dim> const& position) {
		return move(root(), data, position);
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
		return const_cast<Octree<Data, Node, Dim>*>(this)->find(
			start, position);
	}
	
	ConstNodeIterator find(
			Vector<Dim> const& position) const {
		return find(root(), position);
	}
	
	
	
	NodeIterator find(
			ConstNodeIterator start,
			ConstDataIterator data) {
		auto node = start;
		
		std::size_t dataIndex = data - _data.begin();
		std::size_t lower = node->_dataIndex;
		std::size_t upper = lower + node->_dataCount;
		
		bool contains = dataIndex >= lower && dataIndex < upper;
		
		if (contains && !node->_hasChildren) {
			return node;
		}
		else if (contains && node->_hasChildren) {
			for (std::size_t index = 0; index < (1 << Dim); ++index) {
				auto child = node + node->_childIndices[index];
				std::size_t lower = child->_dataIndex;
				std::size_t upper = lower + child->_dataCount;
				if (dataIndex >= lower && dataIndex < upper) {
					return find(child, data);
				}
			}
		}
		else if (node->_hasParent) {
			auto parent = node + node->_parentIndex;
			return find(parent, data);
		}
		return _nodes.end();
	}
	
	NodeIterator find(
			ConstDataIterator data) {
		return find(root(), data);
	}
	
	ConstNodeIterator find(
			ConstNodeIterator start,
			ConstDataIterator data) const {
		return const_cast<Octree<Data, Node, Dim>*>(this)->find(
			start, data);
	}
	
	ConstNodeIterator find(
			ConstDataIterator data) const {
		return find(root(), data);
	}
	
};

}

#endif

