#ifndef __NBODY_OCTREE_H_
#define __NBODY_OCTREE_H_

#include <type_traits>

#include "tensor.h"

namespace nbody {

template<typename T, std::size_t Dim>
class OctreeData {
	
public:
	
	T data;
	OctreeNode<T, Dim>* parent;
	Vector<Dim> position;
	
};

template<typename T, std::size_t Dim>
class OctreeNode {
	
private:
	
	OctreeNode<T, Dim>* _parent;
	OctreeNode<T, Dim>* _children[1 << Dim];
	
	Vector<Dim> _dimensions;
	Vector<Dim> _position;
	
	std::vector<OctreeData<T, Dim> > _data;
	
public:
	
	OctreeNode<T, Dim>* parent() final;
	OctreeNode<T, Dim>* children() final;
	OctreeData<T, Dim>* data() final;
	
	/**
	 * \brief Adds a new piece of data to the OctreeNode.
	 * 
	 * This function will attempt to place the data within this OctreeNode if
	 * there is room. Otherwise, it will split this OctreeNode, and distribute
	 * the data among the children. If the position is out of range for this
	 * OctreeNode, then this function will check the parents of this OctreeNode
	 * until a suitable position can be found.
	 * 
	 * \param data the data to be added
	 * \param position the position at which the data should be added
	 * \return a pointer to the data once it has been copied to the OctreeNode,
	 *         or a null pointer if data could not be added
	 */
	OctreeData<T, Dim>* add(T data, Vector<Dim> position) final;
	
	/**
	 * \brief Removes a piece of data from the OctreeNode.
	 * 
	 * The OctreeNode will search both itself and all of its children for the
	 * piece of data. If it is found, it will be removed.
	 * 
	 * \param data the data that should be removed
	 */
	void remove(OctreeData<T, Dim>* data) final;
	
	/**
	 * \brief Moves an existing piece of data from one position in the
	 * OctreeNode to another. If the new position is out of the range of this
	 * OctreeNode, then the parent of this node will be checked until a
	 * suitable position can be found.
	 * 
	 * \param data a pointer to the data that should be moved
	 * \param position the new position of the data
	 * \return a pointer to the new value of the data, or a null pointer of the
	 *         data could not be moved
	 */
	OctreeData<T, Dim>* move(OctreeData<T, Dim>* data,
	                         Vector<Dim> position) final;
	
	template<typename T, std::size_t Dim, typename N>
	friend class Octree;
	
};

template<typename T, std::size_t Dim = 3, typename N = OctreeNode<T, Dim> >
class Octree final {
	
	static_assert(std::is_base_of<OctreeNode<T, Dim>, N>::value,
	              "N must derive from OctreeNode");
	
private:
	
	N* _root;
	
public:
	
	
	/**
	 * \brief Returns the root OctreeNode of the entire Octree.
	 */
	N* root();
	N const* root() const;
	
	/**
	 * \brief Adds a new piece of data to the Octree.
	 * 
	 * This function is equivalent to calling the OctreeNode::add function on
	 * the root node of this Octree.
	 * 
	 * \param data the data to be added
	 * \param position the position at which the data should be added
	 * \return a pointer to the data once it has been copied to the Octree
	 */
	OctreeData<T, Dim>* add(T data, Vector<Dim> position);
	
	/**
	 * \brief Removes a piece of data from the Octree.
	 * 
	 * This function is equivalent to calling the OctreeNode::remove function
	 * on the root node of this Octree.
	 * 
	 * \param data a pointer to the data that should be removed
	 */
	void remove(OctreeData<T, Dim>* data);
	
	/**
	 * \brief Moves an existing piece of data from one position in the Octree
	 * to another.
	 * 
	 * This function is equivalent to calling the OctreeNode::move function on
	 * the root node of this Octree.
	 * 
	 * \param data a pointer to the data that should be moved
	 * \param position the new position of the data
	 * \return a pointer to the new value of the data
	 */
	OctreeData<T, Dim>* move(OctreeData<T, Dim>* data,
	                         Vector<Dim> position);
	
	/**
	 * \brief Returns a depth-first iterator over all of the data in the
	 * Octree.
	 */
	DepthIterator depthIterator();
	ConstDepthIterator depthIterator() const;
	
	/**
	 * \brief Returns a breadth-first iterator over all of the data in the
	 * Octree.
	 */
	BreadthIterator breadthIterator();
	ConstBreadthIterator breadthIterator() const;
	
	/**
	 * \brief Returns a depth-first iterator over all of the nodes in the
	 * Octree.
	 */
	NodeDepthIterator nodeDepthIterator();
	ConstNodeDepthIterator nodeDepthIterator() const;
	
	/**
	 * \brief Returns a breadth-first iterator over all of the nodes in the
	 * Octree.
	 */
	NodeBreadthIterator nodeBreadthIterator();
	ConstNodeBreadthIterator nodeBreadthIterator() const;
	
};

}

#endif

