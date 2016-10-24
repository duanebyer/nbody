#ifndef __NBODY_OCTREE_H_
#define __NBODY_OCTREE_H_

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
 * \tparam T the type of data stored at the leaves of the Octree
 * \tparam N the type of data stored at the nodes of the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename T, typename N, std::size_t Dim>
class OctreeData final {
	
	friend class OctreeNode<T, N, Dim>;
	
private:
	
	OctreeNode<T, N, Dim>* _parent;
	T _data;
	Vector<Dim> _position;
	
	OctreeData(OctreeNode<T, N, Dim>* parent,
	           T data,
	           Vector<Dim> position) :
	           _parent(parent),
	           _data(data),
	           _position(position) {
	}
	
public:
	
	OctreeNode<T, N, Dim>& parent() {
		return *_parent;
	}
	OctreeNode<T, N, Dim> const& parent() const {
		return *_parent;
	}
	
	T& data() {
		return _data;
	}
	T const& data() const {
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
	
	// The parent of this node.
	OctreeNode<T, N, Dim>* _parent;
	// The children of this node. They are dynamically allocated and managed by
	// this node.
	OctreeNode<T, N, Dim>* _children;
	// The location of the data of this node. Because the data is stored in
	// dynamic memory, a direct pointer cannot be stored.
	std::size_t _data;
	
	// Whether this node has any data.
	bool _hasData;
	
	// The section of space that this node encompasses.
	Vector<Dim> _position;
	Vector<Dim> _dimensions;
	
	OctreeNode() : _octree(NULL),
	               _parent(NULL),
	               _children(NULL),
	               _hasData(false),
	               _position(),
	               _dimensions() {
	}
	
	OctreeNode(OctreeNode<T, N, Dim> const&) = delete;
	void operator=(OctreeNode<T, N, Dim> const&) = delete;
	
	// Splits this node into four new subnodes, and assigns this node's data to
	// the appropriate subnode. This method is UNSAFE: it does not check to see
	// if this node already has children.
	void split() {
		// First, create the children and spatially position them so that they
		// cover the hypercube defined by this node.
		_children = new OctreeNode<T, N, Dim>[1 << Dim];
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
			OctreeNode<T, N, Dim>& child = _children[index];
			child._octree = _octree;
			child._parent = this;
			child._dimensions = _dimensions / 2.0;
			child._position = _position;
			
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				if ((index >> dim) & 1) {
					child._position[dim] += child._dimensions[dim];
				}
			}
		}
		
		// Then, find which subnode the data belongs in, and add it to the
		// appropriate child.
		if (_hasData) {
			Vector<Dim> dataPos = _octree->_data[_data].position();
			std::size_t dataIndex = 0;
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				if (dataPos[dim] - _position[dim] >= _dimensions[dim] / 2.0) {
					dataIndex += (1 << dim);
				}
			}
			_children[dataIndex]._data = _data;
			_children[dataIndex]._hasData = true;
			_hasData = false;
			_octree->_data[_data]._parent = &_children[dataIndex];
		}
	}
	
	// Merges all of this node's subnodes together, absorbing a single data
	// point contained within the subnodes into this node. This method is
	// UNSAFE: it does not take into account grandchildren.
	void merge() {
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
			if (_children[index]._hasData) {
				_data = _children[index]._data;
				_hasData = true;
				_children[index]._hasData = false;
				_octree->_data[_data]._parent = &_children[index];
				break;
			}
		}
		delete [] _children;
		_children = NULL;
	}
	
	// Adds a piece of data to this node. If this node already contains data,
	// then this node is split into subnodes and the data is added to the
	// appropriate subnode.
	OctreeNode<T, N, Dim>* add(std::size_t data) {
		if (!_hasData && _children == NULL) {
			_hasData = true;
			_data = data;
			_octree->_data[_data]._parent = this;
			return this;
		}
		else {
			if (_children == NULL) {
				split();
			}
			Vector<Dim> dataPos = _octree->_data[dataIndex].position();
			std::size_t dataIndex = 0;
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				if (dataPos[dim] - _position[dim] >= _dimensions[dim] / 2.0) {
					dataIndex += (1 << dim);
				}
			}
			return _children[dataIndex].add(data);
		}
	}
	
	// Checks if a node should be merged with its siblings. Recursively calls
	// itself on its parent after a successful merge. This result is the first
	// node in the chain that does not get merged with its siblings.
	OctreeNode<T, N, Dim>* collapse() {
		OctreeNode<T, N, Dim>* result = this;
		OctreeNode<T, N, Dim>* parent = _parent;
		if (parent != NULL) {
			bool shouldMerge = true;
			for (std::size_t index = 0; index < (1 << Dim); ++index) {
				OctreeNode<T, N, Dim>& child = parent->_children[index];
				shouldMerge = shouldMerge && !child._hasData;
			}
			if (shouldMerge) {
				parent->merge();
				result = parent->collapse();
			}
		}
		return result;
	}
	
	// Removes any data that this node stores. Then, collapses this node.
	OctreeNode<T, N, Dim>* remove() {
		_hasData = false;
		_octree->_data[_data]._parent = NULL;
		return collapse();
	}
	
public:
	
	~OctreeNode() {
		if (_children != NULL) {
			delete [] _children;
			_children = NULL;
		}
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
	
	// A list storing all of the actual data. This is so that all of the raw
	// data can be stored contigiously in memory.
	std::vector<OctreeData<T, Dim> > _data;
	// The root of the octree. The root contains dynamically allocated data for
	// the subnodes of the octree.
	OctreeNode<T, N, Dim> _root;
	
public:
	
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
	 * \return a pointer to the OctreeNode that now stores the data
	 */
	OctreeNode<T, N, Dim>* add(T data, Vector<Dim> position);
	
	/**
	 * \brief Removes a piece of data from the Octree.
	 * 
	 * \param data a pointer to the data that should be removed
	 * \return a pointer to the OctreeNode that stored the data
	 */
	OctreeNode<T, N, Dim>* remove(OctreeData<T, Dim>* data);
	
	/**
	 * \brief Moves an existing piece of data from one position in the Octree
	 * to another.
	 * 
	 * \param data a pointer to the data that should be moved
	 * \param position the new position of the data
	 * \return a pair containing the old OctreeNode and the new OctreeNode
	 */
	std::pair<OctreeNode<T, N, Dim>*, std::pair<OctreeNode<T, N, Dim> >
	move(OctreeData<T, Dim>* data,
	     Vector<Dim> position);
	
	OctreeData<T, Dim>* data() {
		return _data;
	}
	
	OctreeData<T, Dim> const* data() const {
		return _data;
	}
	
};

}

#endif

