#ifndef __NBODY_OCTREE_H_
#define __NBODY_OCTREE_H_

#include <cstddef>
#include <type_traits>
#include <vector>

#include "tensor.h"

namespace nbody {

/**
 * \brief Wraps a piece of data together with a position so that it can be
 * stored at the leaf of an Octree.
 * 
 * \tparam T the type of data stored at the leaves of the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename T, std::size_t Dim>
class OctreeData final {
	
private:
	
	Vector<Dim> _position;
	
	OctreeData(Vector<Dim> position,
	           T data) :
	           _position(position),
	           data(data) {
	}
	
public:
	
	T data;
	
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
 * \tparam T the type of data stored at the leaves of the Octree
 * \tparam N the type of data stored at the nodes of the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename T, typename N, std::size_t Dim>
class OctreeNode final {
	
	friend class Octree<T, N, Dim>;
	
private:
	
	// All OctreeNodes are guaranteed to be stored contigiously.
	
	// The octree that contains this node.
	Octree<T, N, Dim> const* _octree;
	
	// The location of this node within the octree.
	std::size_t _node;
	// The location of the parent of this node within the octree.
	std::size_t _parent;
	// Whether this node has any children. If it does, then they will be
	// stored starting at the next index within the octree.
	bool _hasChildren;
	// Whether this node has any data.
	bool _hasData;
	// The location of the data of this node.
	std::size_t _data;
	
	// The section of space that this node encompasses.
	Vector<Dim> _position;
	Vector<Dim> _dimensions;
	
	OctreeNode(Octree<T, N, Dim> const* octree,
	           std::size_t parent,
	           Vector<Dim> position,
	           Vector<Dim> dimensions) :
               _octree(octree),
	           _parent(parent),
	           _hasChildren(false),
	           _hasData(false),
	           _data(0),
	           _position(position),
	           _dimensions(dimensions) {
	}
	
public:
	
	bool hasChildren() const {
		return _hasChildren;
	}
	
	bool hasData() const {
		return _hasData;
	}
	
	bool isRoot() const {
		return _node == 0;
	}
	
	OctreeNode<T, Dim>& parent() {
		return _octree._nodes[_parent];
	}
	OctreeNode<T, Dim> const& parent() const {
		return _octree._nodes[_parent];
	}
	
	OctreeNode<T, Dim>* children() {
		return &_octree._nodes[_node + 1];
	}
	OctreeNode<T, Dim> const* children() const {
		return &_octree._nodes[_node + 1];
	}
	
	OctreeData<T, Dim>& data() {
		return _octree->_data[_data];
	}
	OctreeData<T, Dim> const& data() const {
		return _octree->_data[_data];
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
 * \tparam T the type of data stored at the leaves of the Octree
 * \tparam N the type of data stored at the nodes of the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename T, typename N, std::size_t Dim = 3>
class Octree final {
	
	static_assert(std::is_base_of<OctreeNode<T, Dim>, N>::value,
	              "N must derive from OctreeNode");
	
private:
	
	// A list storing all of the actual data.
	std::vector<OctreeData<T, Dim> > _data;
	// A list storing all of the nodes in depth-first order. The root node is
	// the first one in the list.
	std::vector<OctreeNode<T, N, Dim> > _nodes;
	
	// Takes a node and splits it into a number of sub-nodes. Any data is then
	// moved into the appropriate sub-node.
	void split(std::size_t node) {
		auto nodeIt = _nodes.begin() + node;
		
		Vector<Dim> position = nodeIt->_position;
		Vector<Dim> dimensions = nodeIt->_dimensions / 2.0;
		OctreeData<T, Dim>* data = nodeIt->_data;
		
		nodeIt->_hasChildren = true;
		_nodes.insert(nodeIt,
		              1 << Dim,
		              OctreeData<T, N, Dim>(this, node, position, dimensions));
		nodeIt = _nodes.begin() + node;
		
		// There needs to be one sub-node for each combination of the
		// dimensions. For instance, in 3d, there needs to be a sub-node for
		// the relative positions (-x, -y, -z), (-x, -y, +z), (-x, +y, -z),
		// (-x, +y, +z), and so on. These loops go over a sequence of binary
		// numbers with 'Dim' digits. A digit of '1' indicates that the
		// sub-node should be offset along that dimension, and a digit of '0'
		// indicates that the sub-node should not be offset.
		auto subNodeIt = nodeIt + 1;
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				// If there is a one at the specific digit, then the sub-node
				// needs to be positively shifted along that dimension.
				if ((index >> dim) & 1) {
					subNodeIt->_position[dim] += dimensions[dim];
				}
			}
			++subNodeIt;
		}
		
		// Now the data has to be moved from the parent node to the correct
		// sub-node.
		auto dataNodeIt = nodeIt + 1;
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (data->position[dim] < position[dim] + dimensions[dim]) {
				dataNodeIt += (1 << dim);
			}
		}
		dataNodeIt->_data = nodeIt->_data;
		dataNodeIt->_hasData = true;
		nodeIt->_hasData = false;
	}
	
	// Takes a node and removes all of its sub-nodes. Any data contained within
	// the sub-nodes is moved into the parent node (there should be at most one
	// data point within all of the sub-nodes). This method is not recursive!
	void merge(std::size_t node) {
		auto nodeIt = _nodes.begin() + node;
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
			auto subNodeIt = nodeIt + index;
			if (subNodeIt->_hasData) {
				nodeIt->_data = subNodeIt->_data;
				nodeIt->_hasData = true;
				break;
			}
		}
		nodeIt->_hasChildren = false;
		_nodes.erase(nodeIt, nodeIt + (1 << Dim));
	}
	
public:
	
	using Iterator = decltype(_data)::iterator;
	using ConstIterator = decltype(_data)::const_iterator;
	using NodeIterator = decltype(_nodes)::iterator;
	using ConstNodeIterator = decltype(_nodes)::const_iterator;
	
	Octree(Vector<Dim> position, Vector<Dim> dimensions) : _data(), _nodes() {
		_nodes.push_back(OctreeNode(NULL, position, dimensions));
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
	 * \return a pointer to the data once it has been copied to the Octree
	 */
	OctreeData<T, Dim>* add(T data, Vector<Dim> position);
	
	/**
	 * \brief Removes a piece of data from the Octree.
	 * 
	 * \param data a pointer to the data that should be removed
	 */
	void remove(OctreeData<T, Dim>* data);
	
	/**
	 * \brief Moves an existing piece of data from one position in the Octree
	 * to another.
	 * 
	 * \param data a pointer to the data that should be moved
	 * \param position the new position of the data
	 * \return a pointer to the new value of the data
	 */
	OctreeData<T, Dim>* move(OctreeData<T, Dim>* data,
	                         Vector<Dim> position);
	
	/**
	 * \brief Returns an iterator to the first piece of data.
	 */
	Iterator begin() {
		return _data.begin();
	}
	ConstIterator begin() const {
		return _data.cbegin();
	}
	
	/**
	 * \brief Returns an iterator to the last piece of data.
	 */
	Iterator end() {
		return _data.end();
	}
	ConstIterator end() const {
		return _data.cend();
	}
	
	/**
	 * \brief Returns an iterator to the first OctreeNode in depth-first
	 * order.
	 */
	NodeIterator beginNode() {
		return _nodes.begin();
	}
	ConstNodeIterator beginNode() const {
		return _nodes.cbegin();
	}
	
	/**
	 * \brief Returns an iterator the last OctreeNode in depth-first order.
	 */
	NodeIterator endNode() {
		return _nodes.end();
	}
	ConstNodeIterator endNode() const {
		return _nodes.cend();
	}
	
};

}

#endif

