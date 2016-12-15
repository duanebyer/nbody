#ifndef __NBODY_OCTREE_H_
#define __NBODY_OCTREE_H_

#include <algorithm>
#include <cstddef>
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
	
	// The number of data points that this node contains.
	std::size_t _dataCount;
	// The data points that this node contains are located at this index.
	std::size_t _dataIndex;
	
	// The section of space that this node encompasses.
	Vector<Dim> _position;
	Vector<Dim> _dimensions;
	
	OctreeNode(Octree<Data, Node, Dim> const* octree) :
			_octree(octree),
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
	
	// Divides this node into a set of subnodes and partitions its data between
	// them
	void createChildren() {
		// Create the 2^Dim child nodes inside the parent octree.
		auto begin = _octree->_nodes.begin();
		auto parent = begin + (this - &_octree->_nodes.front());
		auto firstChild = _octree->_nodes.insert(
			parent, 1 << Dim,
			OctreeNode<Data, Node, Dim>(_octree));
		
		// Update the iterators as they have been invalidated.
		begin = _octree->_nodes.begin();
		parent = begin + (this - &_octree->_nodes.front());
		
		// Loop through the new children, and set up their various properties.
		Vector<Dim> childDimensions = _dimensions / 2;
		Vector<Dim> midpoint = _position + childDimensions;
		auto child = firstChild;
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
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
		while (parent->_hasParent) {
			std::size_t siblingIndex = parent->_siblingIndex;
			parent += parent->_parentIndex;
			while (++siblingIndex <= (1 << Dim)) {
				parent->_childIndices[siblingIndex] += (1 << Dim);
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
			moveData(data, firstChild + childIndex);
		}
	}
	
	// Destroys the subnodes of this node and takes their data into this node.
	void destroyChildren() {
		// Determine how many children, grandchildren, great-grandchildren, ...
		// of this node.
		std::size_t numDescendants = _childIndices[1 << Dim];
		auto begin = _octree->_nodes.begin();
		auto parent = begin + (this - &_octree->_nodes.front());
		
		// Reassign all of the data to this node.
		std::size_t lastDataIndex = (parent + numDescendants)->_dataIndex;
		_dataCount = lastDataIndex - _dataIndex;
		
		// Destroy the subnodes and update iterators.
		_octree->_nodes->erase(parent + 1, parent + 1 + numDescendants);
		begin = _octree->_nodes.begin();
		parent = begin + (this - &_octree->nodes.front());
		
		// Go through the parent, grandparent, great-grandparent, ... of this
		// node and update their child indices.
		while (parent->_hasParent) {
			std::size_t siblingIndex = parent->_siblingIndex;
			parent += parent->_parentIndex;
			while (++siblingIndex <= (1 << Dim)) {
				parent->_childIndices[siblingIndex] -= numDescendants;
			}
		}
	}
	
	// Creates a parent node that contains this node.
	void createParent() {
	}
	
	// Destroys all ancestors of this node and takes their data into this node.
	void destroyParent() {
	}
	
	// Adds a piece of data to the octree, starting the search for the correct
	// node at this node.
	Octree<Node, Data, Dim>::DataIterator insertData(
			OctreeData<Data, Dim> const& data) {
		
		Vector<Dim> position = data.position();
		bool contains = contains(position);
		bool canHoldData =
			!_hasChildren &&
			_dataCount < _octree->nodeCapacity;
		if (contains && canHoldData) {
			return insertDataAt(octreeData);
		}
		else if (contains && !canHoldData) {
			return insertDataBelow(octreeData);
		}
		else {
			return insertDataAbove(octreeData);
		}
	}
	
	// Adds a piece of data to this node.
	Octree<Node, Data, Dim>::DataIterator insertDataAt(
			OctreeData<Data, Dim> const& data) {
		// Add the data to the master list of data in the octree and update
		// internal variables.
		auto begin = _octree->_data.begin();
		auto newData = _octree->_data.insert(begin + _dataIndex + _dataCount);
		++_dataCount;
		
		// Loop through the rest of the nodes and increment their data indices
		// so that they still refer to the correct location in the data vector.
		auto begin = _octree->_nodes.begin();
		auto end = _octree->_nodes.end();
		auto node = begin + (this - &_octree->nodes.first());
		while (++node != end) {
			++node->_dataIndex;
		}
		
		return newData;
	}
	
	// Adds a data point to a child of this node, splitting this node into
	// subnodes if necessary.
	Octree<Data, Node, Dim>::DataIterator insertDataBelow(
			OctreeData<Data, Dim> const& data) {
		if (!_hasChildren) {
			createChildren();
		}
		
		auto begin = _octree->_nodes.begin();
		auto node = begin + (this - &_octree->_nodes.first());
		
		// Determine which child the data should be inserted to.
		Vector<Dim> dataPosition = data.position();
		Vector<Dim> position = _position;
		Vector<Dim> midpoint = position + _dimensions / 2;
		std::size_t childIndex = 0;
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (dataPosition[dim] >= midpoint[dim]) {
				childIndex += (1 << Dim);
			}
		}
		
		// Insert the data into the child.
		auto child = node + childIndex;
		return child->insertData(data);
	}
	
	// Adds a data point to a parent of this node, creating a new node to act
	// as a parent if necessary.
	Octree<Data, Node, Dim>::DataIterator insertDataAbove(
			OctreeData<Data, Dim> const& data) {
		if (!_hasParent) {
			createParent();
		}
		
		// Insert the data into the parent.
		auto begin = _octree->_nodes.begin();
		auto node = begin + (this - &_octree->_nodes.first());
		auto parent = node + _parentIndex;
		parent->insertData(data);
	}
	
	// Removes a piece of data from this node.
	Octree<Node, Data, Dim>::DataIterator eraseData(
			Octree<Node, Data, Dim>::DataIterator data) {
	}
	
	// Moves a piece of data from this node to another one.
	Octree<Node, Data, Dim>::DataIterator moveData(
			Octree<Node, Data, Dim>::DataIterator data,
			Octree<Node, Data, Dim>::NodeIterator node) {
	}
	
public:
	
	Octree<Data, Node, Dim>::DataIterator insert(
			Data const& data,
			Vector<Dim> const& position) {
		// Create a new data structure containing the information, determine
		// whether it should be inserted above or below, then call the
		// appropriate function.
		OctreeData<Data, Dim> octreeData(data, position);
		return insertData(octreeData);
	}
	
	Octree<Data, Node, Dim>::DataIterator erase(
			Octree<Data, Node, Dim>::DataIterator it) {
	}
	
	Octree<Data, Node, Dim>::DataIterator erase(
			Octree<Data, Node, Dim>::ConstDataIterator it) {
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
		return _children != NULL;
	}
	
	bool hasData() const {
		return _hasData;
	}
	
	bool hasParent() const {
		return _parent != NULL;
	}
	
	OctreeNode<T, Dim>& parent() {
		return *_parent;
	}
	OctreeNode<T, Dim> const& parent() const {
		return *_parent;
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

