#ifndef __NBODY_OCTREE_H_
#define __NBODY_OCTREE_H_

#include <type_traits>

#include "tensor.h"

namespace nbody {

/**
 * \brief Wraps a piece of data together with a position so that it can be
 * stored in an Octree.
 * 
 * \tparam T the type of data stored in the Octree
 * \tparam Dim the dimension of the space that the Octree is embedded in
 */
template<typename T, std::size_t Dim>
class OctreeData {
	
public:
	
	T data;
	OctreeNode<T, Dim>* parent;
	Vector<Dim> position;
	
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
	
	Vector<Dim> _dimensions;
	Vector<Dim> _position;
	
public:
	
	bool isLeaf() const final {
		return _data != NULL;
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
	
	template<typename T, std::size_t Dim, typename N>
	friend class Octree;
	
};

template<typename T, std::size_t Dim = 3, typename N = OctreeNode<T, Dim> >
class Octree final {
	
	static_assert(std::is_base_of<OctreeNode<T, Dim>, N>::value,
	              "N must derive from OctreeNode");
	
private:
	
	// A list storing all of the actual data.
	std::vector<OctreeData<T, Dim> > _data;
	// A list storing all of the nodes in breadth-first order. The root node
	// is the first one in the list.
	std::vector<OctreeNode<T, Dim> > _nodes;
	
public:
	
	/**
	 * \brief Returns the root OctreeNode of the entire Octree.
	 */
	N* root();
	N const* root() const;
	
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
	Iterator begin();
	ConstIterator begin() const;
	
	/**
	 * \brief Returns an iterator to the last piece of data.
	 */
	Iterator end();
	ConstIterator end() const;
	
	/**
	 * \brief Returns an iterator to the first OctreeNode in breadth-first
	 * order.
	 */
	NodeBreadthIterator beginNodeBreadth();
	ConstNodeBreadthIterator beginNodeBreadth() const;
	
	/**
	 * \brief Returns an iterator to the last OctreeNode in breadth-first
	 * order.
	 */
	NodeBreadthIterator endNodeBreadth();
	ConstNodeBreadthIterator endNodeBreadth() const;
	
	/**
	 * \brief Returns an iterator to the first OctreeNode in depth-first
	 * order.
	 */
	NodeDepthIterator beginNodeDepth();
	ConstNodeDepthIterator beginNodeDepth() const;
	
	/**
	 * \brief Returns an iterator the last OctreeNode in depth-first order.
	 */
	NodeDepthIterator endNodeDepth();
	ConstNodeDepthIterator endNodeDepth() const;
	
};

}

#endif

