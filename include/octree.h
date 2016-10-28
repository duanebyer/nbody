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
	
	friend class Octree<T, N, Dim>;
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
	std::size_t _dataIndex;
	
	// Whether this node has any data.
	bool _hasData;
	
	// The section of space that this node encompasses.
	Vector<Dim> _position;
	Vector<Dim> _dimensions;
	
	OctreeNode() :
			_octree(NULL),
			_parent(NULL),
			_children(NULL),
			_hasData(false),
			_position(),
			_dimensions() {
	}
	
	// These operators allow for copying of the node along with all of its
	// structure, but the data contained within the octree is not copied.
	// Because of this, these operators are not public.
	OctreeNode(OctreeNode<T, N, Dim> const& other) :
			_octree(other._octree),
			_parent(other._parent),
			_hasData(other._hasData),
			_dataIndex(other._dataIndex),
			_position(other._position),
			_dimensions(other._dimensions) {
		if (other.hasChildren()) {
			createChildren();
			for (std::size_t index = 0; index < (1 << Dim); ++index) {
				_children[i] = OctreeNode(other._children[i]);
			}
		}
	}
	
	OctreeNode(OctreeNode<T, N, Dim>&& other) :
			_octree(other._octree),
			_parent(other._parent),
			_children(other._children),
			_hasData(other._hasData),
			_dataIndex(other._dataIndex),
			_position(other._position),
			_dimensions(other._dimensions) {
		other._children = NULL;
	}
	
	~OctreeNode() {
		destroyChildren();
	}
	
	OctreeNode<T, N, Dim>& operator=(OctreeNode<T, N, Dim> const& other) {
		_octree = other._octree;
		_parent = other._parent;
		_hasData = other._hasData;
		_dataIndex = other._dataIndex;
		_position = other._position;
		_dimensions = other._dimensions;
		
		destroyChildren();
		if (other.hasChildren()) {
			createChildren();
			for (std::size_t index = 0; index < (1 << Dim); ++index) {
				_children[i] = OctreeNode(other._children[i]);
			}
		}
		return *this;
	}
	
	OctreeNode<T, N, Dim>& operator=(OctreeNode<T, N, Dim>&& other) {
		_octree = other._octree;
		_parent = other._parent;
		_hasData = other._hasData;
		_dataIndex = other._dataIndex;
		_position = other._position;
		_dimensions = other._dimensions;
		
		destroyChildren();
		_children = other._children;
		other._children = NULL;
		return *this;
	}
	
	
	// Creates the dynamic array that stores the children of this node.
	void createChildren() {
		// Create the children and spatially position them so that they cover
		// the hypercube defined by this node.
		delete [] _children;
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
	}
	
	// Cleans up the dynamic array that stores the children of this node.
	void destroyChildren() {
		delete [] _children;
		_children = NULL;
	}
	
	// Returns a pointer to the child that contains the specified position.
	OctreeNode<T, N, Dim>* getChildByPosition(Vector<Dim> position) {
		OctreeNode<T, N, Dim>* child = _children;
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (dataPos[dim] - _position[dim] >= _dimensions[dim] / 2.0) {
				child += (1 << dim);
			}
		}
		return child;
	}
	
	// Adds a piece of data to this node.
	void addData(OctreeData<T, N, Dim>* data) {
		_dataIndex = data - _octree->_data;
		_hasData = true;
		data->_parent = this;
	}
	
	// Removes the data from this node.
	void removeData() {
		_hasData = false;
		_octree->_data[_dataIndex]->_parent = NULL;
	}
	
	// Takes the data stored by this node, and moves it into another node while
	// removing it from this node.
	void transferData(OctreeNode<T, N, Dim>* other) {
		other->_hasData = _hasData;
		other->_dataIndex = _dataIndex;
		if (other->_hasData) {
			_octree->_data[_dataIndex]->_parent = other;
		}
		_hasData = false;
	}
	
	// Takes the data stored by this node and moves it to the appropriate
	// child.
	void transferDataChild() {
		Vector<Dim> dataPos = _octree->_data[_dataIndex].position();
		transferData(getChildByPosition(dataPos));
	}
	
	// Takes the data stored by this node and moves it to the parent.
	void transferDataParent() {
		transferData(_parent);
	}
	
	// The following functions are the ones which should actually be used by
	// the Octree class in almost all cases.
	
	// Adds a piece of data to this node. If this node already contains data,
	// then this node given children. If this node has children, then the data
	// is sent to the appropriate child.
	OctreeNode<T, N, Dim>* insert(OctreeData<T, N, Dim>* data) {
		if (hasData()) {
			createChildren();
		}
		if (hasChildren()) {
			return getChildByPosition(data->position())->insert(data);
		}
		else {
			addData(data);
			return this;
		}
	}
	
	// Checks if a node should be merged with its siblings. Recursively calls
	// itself on its parent after a successful merge. This result is the first
	// node in the chain that does not get merged with its siblings.
	OctreeNode<T, N, Dim>* collapse() {
		if (_parent != NULL) {
			std::size_t numData = 0;
			OctreeNode<T, N, Dim>* dataChild = _parent->_children;
			for (std::size_t index = 0; index < (1 << Dim); ++index) {
				OctreeNode<T, N, Dim>* child = _parent->_children + index;
				numData += child->_hasData;
				if (numData == 1) {
					dataChild = child;
				}
			}
			if (numData <= 1) {
				dataChild->transferData(parent);
				parent->destroyChildren();
				return parent->collapse();
			}
		}
		
		return this;
	}
	
	// Removes the data from this node, then collapses the node.
	OctreeNode<T, N, Dim>* erase() {
		removeData();
		return collapse();
	}
	
	// Checks whether a point is contained within this node.
	bool contains(Vector<Dim> point) {
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (point[dim] < _position[dim] ||
			    point[dim] >=  _position[dim] + _dimensions[dim]) {
				return false;
			}
		}
		return true;
	}
	
public:
	
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
	std::vector<OctreeData<T, N, Dim> > _data;
	// The root of the octree. The root contains dynamically allocated data for
	// the subnodes of the octree.
	OctreeNode<T, N, Dim> _root;
	
public:
	
	using Iterator = decltype(_data)::iterator;
	using ConstIterator = decltype(_data)::const_iterator;
	
	Octree(Vector<Dim> position, Vector<Dim> dimensions) : _data(), _nodes() {
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

