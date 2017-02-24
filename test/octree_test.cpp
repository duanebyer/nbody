#define BOOST_TEST_MODULE OctreeTest

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "octree.h"
#include "tensor.h"

using namespace nbody;

struct Leaf;
struct Node;

using TestOctree = Octree<Leaf, Node, 3>;
using TestQuadtree = Octree<Leaf, Node, 2>;

struct Leaf {
	std::size_t data;
	explicit Leaf(int data) : data(data) {
	}
	bool operator==(Leaf const& other) const {
		return data == other.data;
	}
	bool operator!=(Leaf const& other) const {
		return data != other.data;
	}
};

struct Node {
	std::size_t data;
	explicit Node(int data = 0) : data(data) {
	}
	bool operator==(Node const& other) const {
		return data == other.data;
	}
	bool operator!=(Node const& other) const {
		return data != other.data;
	}
};

enum class CheckOctreeResult {
	Success,
	RootHasParent,
	LeafDuplicate,
	LeafMissing,
	DepthIncorrect,
	LeafOutOfBounds,
	NodeOverCapacity,
	NodeOverDepth,
	NodeUnderCapacity,
	ChildLeafDuplicate,
	ChildLeafMissing,
	ChildParentMismatch,
	LeafNotInChild,
	LeafNotInParent,
	ChildCountMismatch,
};

// Takes an octree and a list of leafs that should be contained within the
// octree. Returns whether the structure of the octree is correct for the given
// points.
template<std::size_t Dim>
CheckOctreeResult checkOctree(
		Octree<Leaf, Node, Dim> const& octree,
		std::vector<std::pair<Leaf, Vector<Dim> > > allLeafPairs) {
	// Create a stack storing the points that belong to the current node.
	std::vector<std::vector<std::pair<Leaf, Vector<Dim> > > > stack;
	stack.push_back(allLeafPairs);
	
	// Check that the root node has no parent.
	if (octree.nodes().begin().hasParent()) {
		return CheckOctreeResult::RootHasParent;
	}
	
	// Loop through all of the nodes.
	for (
			auto node = octree.nodes().begin();
			node != octree.nodes().end();
			++node) {
		// Check that the current node has depth of +1 from its parent.
		if (node.depth() !=
				(node.hasParent() ? node.parent().depth() + 1 : 0)) {
			return CheckOctreeResult::DepthIncorrect;
		}
		
		// Take the top of the stack, and check whether each of the
		// leaf-position pairs are within the dimensions.
		std::vector<std::pair<Leaf, Vector<Dim> > > leafPairs(stack.back());
		stack.pop_back();
		
		if (leafPairs.size() > node.leafs().size()) {
			return CheckOctreeResult::LeafDuplicate;
		}
		if (leafPairs.size() < node.leafs().size()) {
			return CheckOctreeResult::LeafMissing;
		}
		
		for (auto leafPair : leafPairs) {
			auto leaf = std::find(
				node.leafs().begin(),
				node.leafs().end(),
				leafPair.first);
			if (leaf == node.leafs().end()) {
				return CheckOctreeResult::LeafMissing;
			}
			if (leaf.position() != leafPair.second) {
				for (std::size_t dim = 0; dim < Dim; ++dim) {
					Scalar lower = node.position()[dim];
					Scalar upper = lower + node.dimensions()[dim];
					if (!(
							leaf.position()[dim] >= lower &&
							leaf.position()[dim] < upper)) {
						return CheckOctreeResult::LeafOutOfBounds;
					}
				}
			}
		}
		
		if (!node.hasChildren()) {
			// If the node doesn't have children, then make sure that it doesn't
			// have too many leafs and that it isn't too deep.
			if (node.leafs().size() > octree.nodeCapacity()) {
				return CheckOctreeResult::NodeOverCapacity;
			}
			if (node.depth() > octree.maxDepth()) {
				return CheckOctreeResult::NodeOverDepth;
			}
		}
		else {
			// Otherwise, make sure it doens't have too few leafs either.
			if (node.leafs().size() <= octree.nodeCapacity()) {
				return CheckOctreeResult::NodeUnderCapacity;
			}
			for (
					std::size_t childIndex = 0;
					childIndex < (1 << Dim);
					++childIndex) {
				auto child = node.child(childIndex);
				
				// Check that the child's parent is this node.
				if (child.parent() != node) {
					return CheckOctreeResult::ChildParentMismatch;
				}
				// Create a vector to store the leaf-position pairs that belong
				// to the child.
				std::vector<std::pair<Leaf, Vector<Dim> > > childLeafPairs;
				auto lastChildLeaf = std::partition(
					leafPairs.begin(),
					leafPairs.end(),
					[child](std::pair<Leaf, Vector<Dim> > leafPair) {
						auto begin = child.leafs().begin();
						auto end = child.leafs().end();
						return std::find(begin, end, leafPair.first) != end;
					});
				std::copy(
					leafPairs.begin(),
					lastChildLeaf,
					std::back_inserter(childLeafPairs));
				leafPairs.erase(leafPairs.begin(), lastChildLeaf);
				
				// Put the child leaf pairs onto the stack.
				if (childLeafPairs.size() != child.leafs().size()) {
					return CheckOctreeResult::LeafNotInParent;
				}
				stack.push_back(childLeafPairs);
			}
			// Check that each of the leaf-position pairs belonged to at least
			// one of the children.
			if (!leafPairs.empty()) {
				return CheckOctreeResult::LeafNotInChild;
			}
		}
	}
	
	// The stack should be empty, except if one of the nodes didn't have the
	// right number of children.
	if (!stack.empty()) {
		return CheckOctreeResult::ChildCountMismatch;
	}
	
	return CheckOctreeResult::Success;
}

// Insert 8 leafs into an octree, one per octant.
BOOST_AUTO_TEST_CASE(OctreeShallowInsertTest) {
	TestOctree octree(
		{0.0, 0.0, 0.0},
		{1.0, 1.0, 1.0},
		3, 4);
	
	// Put a single leaf in each octant of the octree. Check that after the
	// third insertion, the root node subdivides.
	for (std::size_t index = 0; index < 8; ++index) {
		Vector<3> position = {
			(Scalar) (index >> 0 & 1),
			(Scalar) (index >> 1 & 1),
			(Scalar) (index >> 2 & 1)
		};
		position *= 0.9;
		position += {0.05, 0.05, 0.05};
		octree.insert(Leaf(index), position);
		if (index < 3) {
			BOOST_REQUIRE_MESSAGE(
				octree.nodes().size() == 1,
				"the root node should have no children for " +
				std::to_string(index + 1) + " leafs");
		}
		else {
			BOOST_REQUIRE_MESSAGE(
				octree.nodes().size() == 9,
				"the root node should have children for " +
				std::to_string(index + 1) + " leafs");
		}
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
			leaf->data == index,
			"leaf has data " << leaf->data << ", should be " << index);
	}
}

BOOST_AUTO_TEST_CASE(OctreeCheckTest) {
	TestQuadtree quadtree(
		{0.0, 0.0},
		{16.0, 16.0},
		3, 4);
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
	std::vector<std::pair<Leaf, Vector<2> > > leafPairs;
	for (std::size_t index = 0; index < numLeafs; ++index) {
		Leaf leaf(index);
		Vector<2> position(positions[index]);
		leafPairs.push_back(std::make_pair(leaf, position));
		quadtree.insert(leaf, position);
	}
	CheckOctreeResult result = checkOctree(quadtree, leafPairs);
	BOOST_REQUIRE_MESSAGE(
		result == CheckOctreeResult::Success,
		"octree has invalid form (" + std::to_string((int) result) + ")");
}

// Insert a number of leafs into a quadtree, testing deeper insertion.
BOOST_AUTO_TEST_CASE(OctreeDeepInsertTest) {
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
	
	// Actually insert all of the leafs.
	for (std::size_t index = 0; index < numLeafs; ++index) {
		quadtree.insert(Leaf(index), positions[index]);
	}
	
	// Iterate over both the leafs and the nodes and compare them to the arrays
	// from above.
	for (std::size_t index = 0; index < numLeafs; ++index) {
		std::size_t leafData = quadtree.leafs()[index].data;
		BOOST_REQUIRE_MESSAGE(
			leafData == order[index],
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

BOOST_AUTO_TEST_CASE(OctreeSamePointInsertTest) {
	// Create a quadtree with a maximum depth of 3. This corresponds to 4
	// generations of nodes (since the root node is at a depth of 0).
	TestQuadtree quadtree(
		{0.0, 0.0},
		{1.0, 1.0},
		3, 3);
	
	for (std::size_t index = 0; index < 4; ++index) {
		quadtree.insert(Leaf(index), {1.0 / 16.0, 1.0 / 16.0});
	}
	
	TestQuadtree::NodeIterator bottomNodeIt = quadtree.nodes().begin() + 3;
	BOOST_REQUIRE_MESSAGE(
		!bottomNodeIt.hasChildren(),
		"deepest node shouldn't have children");
	BOOST_REQUIRE_MESSAGE(
		bottomNodeIt.leafs().size() == 4,
		"deepest node should have 4 children");
	
	for (std::size_t index = 0; index < quadtree.leafs().size(); ++index) {
		std::size_t data = bottomNodeIt.leafs()[index].data;
		BOOST_REQUIRE_MESSAGE(
			data == index,
			"node at index " + std::to_string(index) +
			" has data " + std::to_string(data));
	}
}

BOOST_AUTO_TEST_CASE(OctreeShallowEraseTest) {
	TestOctree octree(
		{0.0, 0.0, 0.0},
		{1.0, 1.0, 1.0},
		3, 4);
	
	// Insert a single point into each quadrant.
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
	
	// Remove the points one at a time. Check that the number of nodes decreases
	// correctly as points are removed.
	for (std::size_t index = 8; index-- > 0; ) {
		TestOctree::LeafIterator leaf = octree.leafs().end() - 1;
		BOOST_REQUIRE_MESSAGE(
			leaf->data == index,
			"leaf at index " + std::to_string(index) +
			" has data " + std::to_string(leaf->data));
		octree.erase(leaf);
		if (index <= 3) {
			BOOST_REQUIRE_MESSAGE(
				octree.nodes().size() == 1,
				"the root node should have no children for " +
				std::to_string(index) + " leafs");
		}
		else {
			BOOST_REQUIRE_MESSAGE(
				octree.nodes().size() == 9,
				"the root node should have children for " +
				std::to_string(index) + " leafs");
		}
	}
}

BOOST_AUTO_TEST_CASE(OctreeDeepEraseTest) {
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
	
	// Insert all of the leafs.
	for (std::size_t index = 0; index < numLeafs; ++index) {
		quadtree.insert(Leaf(index), positions[index]);
	}
}

