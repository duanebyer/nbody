#define BOOST_TEST_MODULE OctreeTest

#include "octree.h"
#include "tensor.h"

#include <boost/test/unit_test.hpp>

using namespace nbody;

struct Leaf;
struct Node;

using TestOctree = Octree<Leaf, Node, 3>;

struct Leaf {
	int data;
	explicit Leaf(int data) : data(data) {
	}
};

struct Node {
	int data;
	explicit Node(int data = 0) : data(data) {
	}
};

BOOST_AUTO_TEST_CASE(OctreeInsertionTest) {
	// Create an octree with a node capacity of 3 and a max depth of 4.
	TestOctree octree(
		{0.0, 0.0, 0.0},
		{1.0, 1.0, 1.0},
		3, 4);
	
	// Put a single leaf in each octant of the octree.
	for (std::size_t index = 0; index < 8; ++index) {
		Vector<3> position =
			{(Scalar) (index & 1), (Scalar) (index & 2), (Scalar) (index & 4)};
		position *= 0.9;
		position += {0.05, 0.05, 0.05};
		octree.insert(Leaf(index), position);
	}
	
	TestOctree::NodeIterator root = octree.nodes().begin();
	
	BOOST_REQUIRE_MESSAGE(
		root.hasChildren(),
		"root should have child nodes");
	
	TestOctree::NodeRange children = root.children();
	
	BOOST_REQUIRE_MESSAGE(
		children.size() == 8,
		"root should have 8 children");
	
	for (std::size_t index = 0; index < 8; ++index) {
		TestOctree::NodeIterator child = children.begin() + index;
		TestOctree::LeafRange leafs = child.leafs();
		TestOctree::LeafIterator leaf = leafs.begin();
		BOOST_REQUIRE_MESSAGE(
			leafs.size() == 1,
			"child " << index << " should have 1 leaf");
		BOOST_REQUIRE_MESSAGE(
			(std::size_t) leaf->data == index,
			"leaf has data " << leaf->data << ", should be " << index);
	}
}

