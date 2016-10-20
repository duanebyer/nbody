#ifndef __NBODY_OCTREE_H_
#define __NBODY_OCTREE_H_

#include <type_traits>
#include <vector>

#include "tensor.h"

namespace nbody {

/**
 * \brief Wraps a piece of data together with a position so that it can be
 * stored in an Octree.
 * 
 * This class also contains a reference to the OctreeNode that contains this
 * data. Neither the position nor the reference to the containing OctreeNode
 * can be modified.
 * 
 * \tparam T the type of data stored in the Octree
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
 * Each OctreeNode either contains a certain number of children, depending on
 * the dimension of the space, or a single data point.
 * 
 * This class can be safely derived from. This should be done if data should
 * be associated with each individual OctreeNode. For instance, if each
 * OctreeNode should store the center of mass of its children, then a custom
 * extension of this class could override certain methods to track changes in
 * the center of mass.
 * 
 * \tparam T the type of data stored in the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename T, std::size_t Dim>
class OctreeNode {
	
private:
	
	OctreeNode<T, Dim>* _parent;
	OctreeNode<T, Dim>* _children[1 << Dim];
	OctreeData<T, Dim>* _data;
	
	Vector<Dim> _position;
	Vector<Dim> _dimensions;
	
	OctreeNode() = delete;
	
protected:
	
	/**
	 * \brief Any deriving class must have a constructor of this form within
	 * the protected scope that calls the super constructor.
	 * 
	 * This constructor should not be exposed. OctreeNode%s should not be
	 * created except through the Octree class.
	 */
	OctreeNode(OctreeNode<T, Dim>* parent,
	           Vector<Dim> position,
	           Vector<Dim> dimensions) :
	           _parent(parent),
	           _children(),
	           _data(NULL),
	           _position(position),
	           _dimensions(dimensions) {
	}
	
public:
	
	bool isLeaf() const final {
		return _data != NULL;
	}
	
	bool isParent() const final {
		return _children[0] != NULL;
	}
	
	OctreeNode<T, Dim>* parent() final {
		return _parent;
	}
	OctreeNode<T, Dim> const* parent() const final {
		return _parent;
	}
	
	OctreeNode<T, Dim>* children() final {
		return _children;
	}
	OctreeNode<T, Dim> const* children() const final {
		return _children;
	}
	
	OctreeData<T, Dim>* data() final {
		return _data;
	}
	OctreeData<T, Dim> const* data() const final {
		return _data;
	}
	
	Vector<Dim> position() const final {
		return _position;
	}
	
	Vector<Dim> dimensions() const final {
		return _dimensions;
	}
	
	template<typename N>
	friend class Octree<T, Dim, N>;
	
};

template<typename T, std::size_t Dim = 3, typename N = OctreeNode<T, Dim> >
class Octree final {
	
	static_assert(std::is_base_of<OctreeNode<T, Dim>, N>::value,
	              "N must derive from OctreeNode");
	
private:
	
	// A list storing all of the actual data.
	std::vector<OctreeData<T, Dim> > _data;
	// A list storing all of the nodes in depth-first order. The root node is
	// the first one in the list.
	std::vector<N> _nodes;
	
	// Takes a node and splits it into a number of sub-nodes. Any data is then
	// moved into the appropriate sub-node.
	void split(N* node) {
		std::size_t nodeOffset = node - &_data[0];
		Vector<Dim> position = node->_position;
		Vector<Dim> dimensions = node->_dimensions / 2.0;
		OctreeData<T, Dim>* data = node->_data;
		
		_nodes.insert(nodeIt,
		              1 << Dim,
		              N(node, position, dimensions));
		
		auto nodeIt = _nodes.begin() + nodeOffset;
		
		// There needs to be one sub-node for each combination of the
		// dimensions. For instance, in 3d, there needs to be a sub-node for
		// the relative positions (-x, -y, -z), (-x, -y, +z), (-x, +y, -z),
		// (-x, +y, +z), and so on. These loops go over a sequence of binary
		// numbers with 'Dim' digits. A digit of '1' indicates that the
		// sub-node should be offset along that dimension, and a digit of '0'
		// indicates that the sub-node should not be offset.
		auto subNodeIt = nodeIt;
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
		std::size_t dataIndex = 0;
		for (std::size_t dim = 0; dim < Dim; ++dim) {
			if (data->position[dim] < position[dim] + dimensions[dim]) {
				dataIndex += (1 << dim);
			}
		}
		std::swap(nodeIt->_data, (nodeIt + dataIndex)->_data);
	}
	
	// Takes a node and removes all of its sub-nodes. Any data contained within
	// the sub-nodes is moved into the parent node (there should be at most one
	// data point within all of the sub-nodes). This method is not recursive!
	void merge(N* node) {
		std::size_t nodeOffset = node - &data[0];
		auto nodeIt = _nodes.begin() + nodeOffset;
		for (std::size_t index = 0; index < (1 << Dim); ++index) {
			OctreeData<T, Dim>* subData = node->_children[index]->_data;
			if (subData != NULL) {
				node->_data = subData;
				break;
			}
		}
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
	N& root() {
		return _nodes[0];
	}
	N const& root() const {
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

