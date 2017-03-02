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
	LeafPositionMismatch,
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

// Takes an octree and a list of leaf-position pairs that should be contained
// within it. Checks the structure of the octree to make sure that the leafs
// are located at appropriate locations within the octree.
template<std::size_t Dim>
CheckOctreeResult checkOctree(
		Octree<Leaf, Node, Dim> const& octree,
		std::vector<std::pair<Leaf, Vector<Dim> > > allLeafPairs);

// Inserts a set of points into an octree, and checks each time a point is
// inserted that the octree is still valid. Takes in a set of points to insert,
// as well as the set of points already contained in the octree. The octree and
// the set of points contained within it are both passed by reference, and will
// be updated to reflect the insertion.
template<std::size_t Dim>
void testInsertOctree(
		Octree<Leaf, Node, Dim>& octree,
		std::vector<std::pair<Leaf, Vector<Dim> > >& leafPairs,
		std::vector<std::pair<Leaf, Vector<Dim> > > insertLeafPairs);

// Erases a set of points from an octree, and checks each time a point is erased
// that the octree is still valid. Takes in a set of points to erase, as well as
// a set of points already in the octree. The octree and the set of points
// contained within it are both passed by reference, and will be updated to
// reflect the erasing.
template<std::size_t Dim>
void testEraseOctree(
		Octree<Leaf, Node, Dim>& octree,
		std::vector<std::pair<Leaf, Vector<Dim> > >& leafPairs,
		std::vector<std::pair<Leaf, Vector<Dim> > > eraseLeafPairs);

// Insert 8 leafs into an octree, one per octant, then removes them in the same
// order.
BOOST_AUTO_TEST_CASE(OctreeShallowInsertEraseTest) {
	TestOctree octree(
		{0.0, 0.0, 0.0},
		{1.0, 1.0, 1.0},
		3, 4);
	
	// Put a single leaf in each quadrant.
	std::vector<std::pair<Leaf, Vector<3> > > leafPairs;
	std::vector<std::pair<Leaf, Vector<3> > > insertLeafPairs;
	for (std::size_t index = 0; index < 8; ++index) {
		Vector<3> position = {
			(Scalar) (index >> 0 & 1),
			(Scalar) (index >> 1 & 1),
			(Scalar) (index >> 2 & 1)
		};
		position *= 0.9;
		position += {0.05, 0.05, 0.05};
		Leaf leaf(index);
		insertLeafPairs.push_back({leaf, position});
	}
	
	// Insert the points, then erase them.
	testInsertOctree(octree, leafPairs, insertLeafPairs);
	testEraseOctree(octree, leafPairs, leafPairs);
}

// Creates a simple octree, adds a number of points, and then removes them,
// checking for validity after each step.
BOOST_AUTO_TEST_CASE(OctreeInsertEraseTest) {
	TestQuadtree octree(
		{0.0, 0.0},
		{16.0, 16.0},
		3, 4);
	
	// A list of points for a simple octree for testing purposes.
	Vector<2> points[] = {
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
	std::size_t numPoints = sizeof(points) / sizeof(points[0]);
	
	std::vector<std::pair<Leaf, Vector<2> > > leafPairs;
	std::vector<std::pair<Leaf, Vector<2> > > insertLeafPairs;
	for (std::size_t index = 0; index < numPoints; ++index) {
		Vector<2> position = points[index];
		Leaf leaf(index);
		insertLeafPairs.push_back({leaf, position});
	}
	
	// Insert the points, then erase them.
	testInsertOctree(octree, leafPairs, insertLeafPairs);
	testEraseOctree(octree, leafPairs, leafPairs);
}

BOOST_AUTO_TEST_CASE(OctreeSamePointInsertEraseTest) {
	// Create a quadtree with a maximum depth of 3. This corresponds to 4
	// generations of nodes (since the root node is at a depth of 0).
	TestQuadtree octree(
		{0.0, 0.0},
		{1.0, 1.0},
		3, 3);
	
	std::vector<std::pair<Leaf, Vector<2> > > leafPairs;
	std::vector<std::pair<Leaf, Vector<2> > > insertLeafPairs;
	for (std::size_t index = 0; index < 4; ++index) {
		Leaf leaf(index);
		Vector<2> position = {1.0 / 16.0, 1.0 / 16.0};
		insertLeafPairs.push_back({leaf, position});
	}
	
	// Insert the points, then erase them.
	testInsertOctree(octree, leafPairs, insertLeafPairs);
	testEraseOctree(octree, leafPairs, leafPairs);
}

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
				return CheckOctreeResult::LeafPositionMismatch;
			}
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
		
		if (!node.hasChildren()) {
			int depthSign =
				(node.depth() > octree.maxDepth()) -
				(node.depth() < octree.maxDepth());
			// If the node doesn't have children, then make sure that it doesn't
			// have too many leafs and that it isn't too deep.
			if (depthSign < 0 && node.leafs().size() > octree.nodeCapacity()) {
				return CheckOctreeResult::NodeOverCapacity;
			}
			if (depthSign > 0) {
				return CheckOctreeResult::NodeOverDepth;
			}
		}
		else {
			// Otherwise, make sure it doens't have too few leafs either.
			if (node.leafs().size() <= octree.nodeCapacity()) {
				return CheckOctreeResult::NodeUnderCapacity;
			}
			// Iterate over every child, and add its leafs to the stack (in
			// reverse order so that the children are added to the stack in
			// order).
			for (std::size_t childIndex = (1 << Dim); childIndex-- > 0; ) {
				auto child = node.child(childIndex);
				
				// Check that the child's parent is this node.
				if (child.parent() != node) {
					return CheckOctreeResult::ChildParentMismatch;
				}
				// Create a vector to store the leaf-position pairs that belong
				// to the child.
				std::vector<std::pair<Leaf, Vector<Dim> > > childLeafPairs;
				
				// Remove the child's leafs from leafPairs and put them into
				// childLeafPairs instead.
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

template<std::size_t Dim>
void testInsertOctree(
		Octree<Leaf, Node, Dim>& octree,
		std::vector<std::pair<Leaf, Vector<Dim> > >& leafPairs,
		std::vector<std::pair<Leaf, Vector<Dim> > > insertLeafPairs) {
	// Loop through every leaf pair that is to be inserted.
	while (insertLeafPairs.size() > 0) {
		// Get the next leaf to insert.
		std::pair<Leaf, Vector<Dim> > insertLeafPair = insertLeafPairs.front();
		insertLeafPairs.erase(insertLeafPairs.begin());
		
		// Insert it into both leafPairs and the octree itself.
		leafPairs.push_back(insertLeafPair);
		octree.insert(insertLeafPair.first, insertLeafPair.second);
		
		// Finally, check that the octree is still valid.
		CheckOctreeResult check = checkOctree(octree, leafPairs);
		BOOST_REQUIRE_MESSAGE(
			check == CheckOctreeResult::Success,
			"octree has invalid form (error " +
			std::to_string((int) check) + ") " +
			"when adding leaf index " +
			std::to_string(insertLeafPair.first.data));
	}
}

template<std::size_t Dim>
void testEraseOctree(
		Octree<Leaf, Node, Dim>& octree,
		std::vector<std::pair<Leaf, Vector<Dim> > >& leafPairs,
		std::vector<std::pair<Leaf, Vector<Dim> > > eraseLeafPairs) {
	// Loop through every leaf pair that is to be erased.
	while (eraseLeafPairs.size() > 0) {
		// Get the next leaf to erase.
		std::pair<Leaf, Vector<Dim> > eraseLeafPair = eraseLeafPairs.front();
		eraseLeafPairs.erase(eraseLeafPairs.begin());
		
		// Find the corresponding pair that is already in leafPairs.
		auto octreeLeafPair = std::find(
			leafPairs.begin(),
			leafPairs.end(),
			eraseLeafPair);
		
		// Find the leaf within the octree itself.
		auto leaf = std::find(
			octree.leafs().begin(),
			octree.leafs().end(),
			eraseLeafPair.first);
		
		// Verify that all of the iterators are valid.
		BOOST_REQUIRE_MESSAGE(
			leaf != octree.leafs().end(),
			"couldn't find leaf index " +
			std::to_string(eraseLeafPair.first.data));
		BOOST_REQUIRE_MESSAGE(
			octreeLeafPair != leafPairs.end(),
			"couldn't find leaf pair index " +
			std::to_string(eraseLeafPair.first.data));
		
		// Erase the leafs from the octree, as well as from leafPairs.
		leafPairs.erase(octreeLeafPair);
		octree.erase(leaf);
		
		// Finally, check that the octree is still valid.
		CheckOctreeResult check = checkOctree(octree, leafPairs);
		BOOST_REQUIRE_MESSAGE(
			check == CheckOctreeResult::Success,
			"octree has invalid form (error " +
			std::to_string((int) check) + ") " +
			"when removing leaf index " +
			std::to_string(eraseLeafPair.first.data));
	}
}

