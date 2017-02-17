#define BOOST_TEST_MODULE OctreeTest

#include <boost/test/unit_test.hpp>

#include <string>

#include "octree.h"
#include "tensor.h"

using namespace nbody;

struct Leaf;
struct Node;

using TestOctree = Octree<Leaf, Node, 3>;
using TestQuadtree = Octree<Leaf, Node, 2>;

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

// Insert 8 leafs into an octree, one per octant.
BOOST_AUTO_TEST_CASE(OctreeShallowInsertionTest) {
	TestOctree octree(
		{0.0, 0.0, 0.0},
		{1.0, 1.0, 1.0},
		3, 4);
	
	// Put a single leaf in each octant of the octree, in a specific order.
	for (std::size_t index = 0; index < 8; ++index) {
		Vector<3> position = {
			(Scalar) (index >> 0 & 1),
			(Scalar) (index >> 1 & 1),
			(Scalar) (index >> 2 & 1)
		};
		position *= 0.9;
		position += {0.05, 0.05, 0.05};
		octree.insert(Leaf(index), position);
	}
	
	BOOST_REQUIRE_MESSAGE(
		octree.leafs().size() == 8,
		"root should have 8 leafs");
	
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

// Insert a number of leafs into a quadtree, testing deeper insertion.
BOOST_AUTO_TEST_CASE(OctreeDeepInsertionTest) {
	TestQuadtree quadtree(
		{0.0, 0.0},
		{16.0, 16.0},
		3, 4);
	
	// The positions where the leafs will be placed.
	Vector<2> positions[] = {
		{1,  2},
		{6,  2},
		{6,  6},
		{3,  2},
		{2,  6},
		{14, 6},
		{6,  14},
		{6,  10},
		{2,  10},
		{2,  14},
		
		{10, 6},
		{10, 2},
		{9,  9},
		{15, 1},
		{13, 3},
		{15, 3},
		{13, 1},
		{11, 9},
		{9,  11},
		{11, 11},
		
		{15, 9},
		{15, 13},
		{15, 11},
		{15, 15},
		{13, 9},
		{13, 13},
		{11, 13},
		{9,  13},
		{11, 15},
		{9,  15},
	};
	std::size_t numLeafs = sizeof(positions) / sizeof(positions[0]);
	
	// These arrays are used to verify that the structure of the quadtree is
	// correct.
	
	// The iteration order over the leafs (with depth-first).
	std::size_t order[] = {
		0,  3,  1,  4,  2,  11, 16, 13, 14, 15,
		10, 5,  8,  7,  9,  6,  12, 17, 18, 19,
		20, 22, 24, 27, 26, 29, 28, 21, 23, 25,
	};
	// Which nodes in the depth-first iteration order have children.
	bool nodeHasChildren[] = {
		true,
			true, false, false, false, false,
			true,
				false,
				true, false, false, false, false,
				false,
				false,
			true, false, false, false, false,
			true,
				true, false, false, false, false,
				false,
				true, false, false, false, false,
				false,
	};
	// How many leafs each node should have.
	std::size_t nodeNumLeafs[] = {
		30,
			5, 2, 1, 1, 1,
			7, 1, 4, 1, 1, 1, 1, 1, 1,
			4, 1, 1, 1, 1,
			14, 4, 1, 1, 1, 1, 3, 4, 1, 1, 1, 1, 3,
	};
	
	for (std::size_t index = 0; index < numLeafs; ++index) {
		quadtree.insert(Leaf(index), positions[index]);
	}
	
	// Iterate over both the leafs and the nodes and compare them to the arrays
	// from above.
	for (std::size_t index = 0; index < numLeafs; ++index) {
		int leafData = quadtree.leafs()[index].data;
		BOOST_REQUIRE_MESSAGE(
			(std::size_t) leafData == order[index],
			"leaf at index " + std::to_string(index) +
			" should have data " + std::to_string(order[index]) +
			" instead of data " + std::to_string(leafData));
	}
	
	std::size_t nodeIndex = 0;
	TestQuadtree::NodeIterator nodeIt = quadtree.nodes().begin();
	while (nodeIt != quadtree.nodes().end()) {
		BOOST_REQUIRE_MESSAGE(
			nodeIt.hasChildren() == nodeHasChildren[nodeIndex],
			"node at index " + std::to_string(nodeIndex) + " should" +
			(!nodeHasChildren[nodeIndex] ? " not" : "") + " have children");
		BOOST_REQUIRE_MESSAGE(
			nodeIt.leafs().size() == nodeNumLeafs[nodeIndex],
			"node at index " + std::to_string(nodeIndex) +
			" should have " + std::to_string(nodeNumLeafs[nodeIndex]) +
			" leafs instead of " + std::to_string(nodeIt.leafs().size()) +
			" leafs");
		++nodeIndex;
		++nodeIt;
	}
}

