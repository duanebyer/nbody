#define BOOST_TEST_MODULE OctreeTest

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include <algorithm>
#include <array>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "octree.h"
#include "tensor.h"

namespace bdata = boost::unit_test::data;
using namespace nbody;

struct Leaf;
struct Node;

template<std::size_t Dim>
using LeafPair = std::tuple<Leaf, Vector<Dim> >;

template<std::size_t Dim>
using TestOctree = Octree<Leaf, Node, Dim>;

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

// This is needed so that Boost can print out test messages.
namespace boost {
namespace test_tools {
namespace tt_detail {

template<>
struct print_log_value<Leaf> {
	void operator()(std::ostream& os, Leaf const& leaf) {
		os << "Leaf(" << leaf.data << ")";
	}
};

template<std::size_t Dim>
struct print_log_value<Vector<Dim> > {
	void operator()(std::ostream& os, Vector<Dim> const& vector) {
		os << "<";
		if (Dim != 0) {
			os << vector[0];
		}
		for (std::size_t dim = 1; dim < Dim; ++dim) {
			os << ", " << vector[dim];
		}
		os << ">";
	}
};

template<std::size_t Dim>
struct print_log_value<LeafPair<Dim> > {
	void operator()(std::ostream& os, LeafPair<Dim> const& pair) {
		os << "(";
		print_log_value<Leaf>()(os, std::get<Leaf>(pair));
		os << ", ";
		print_log_value<Vector<Dim> >()(os, std::get<Vector<Dim> >(pair));
		os << ")";
	}
};

template<typename T>
struct print_log_value<std::vector<T> > {
	void operator()(std::ostream& os, std::vector<T> const& vector) {
		os << "{";
		if (!vector.empty()) {
			print_log_value<T>()(os, vector[0]);
		}
		for (std::size_t index = 1; index < vector.size(); ++index) {
			os << ", ";
			print_log_value<T>()(os, vector[index]);
		}
		os << "}";
	}
};

template<std::size_t Dim>
struct print_log_value<TestOctree<Dim> > {
	void operator()(std::ostream& os, TestOctree<Dim> const& octree) {
		os << "Octree(";
		os << "node capacity: " << octree.nodeCapacity() << ", ";
		os << "max depth: " << octree.maxDepth() << ", ";
		os << "position: ";
		print_log_value<Vector<Dim> >()(
			os,
			octree.nodes().begin().position());
		os << ", ";
		os << "dimensions: ";
		print_log_value<Vector<Dim> >()(
			os,
			octree.nodes().begin().dimensions());
		os << ")";
	}
};

}
}
}

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
	ChildParentMismatch,
	LeafNotInChild,
	LeafNotInParent,
	ChildCountMismatch,
};

// A collection of different octrees with various parameters.
static auto const octreeData =
	bdata::make(TestOctree<3>({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 3, 4)) +
	bdata::make(TestOctree<3>({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 3, 0)) +
	bdata::make(TestOctree<3>({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 3, 1)) +
	bdata::make(TestOctree<3>({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 3, 64)) +
	bdata::make(TestOctree<3>({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 1, 64)) +
	bdata::make(TestOctree<3>({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 64, 4)) +
	bdata::make(TestOctree<3>({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 64, 0)) +
	bdata::make(TestOctree<3>({-48.0, -32.0, 8.0}, {+64.0, +128.0, 4.0}, 3, 4));

// A set of leaf pair lists that can be used to construct octrees.
static auto const leafPairsData = bdata::make({
	// Shallow octree with a single point in each octant.
	std::vector<LeafPair<3> > {
		LeafPair<3> {Leaf(0), {4,  4,  4}},
		LeafPair<3> {Leaf(1), {12, 4,  4}},
		LeafPair<3> {Leaf(2), {4,  12, 4}},
		LeafPair<3> {Leaf(3), {12, 12, 4}},
		LeafPair<3> {Leaf(4), {4,  4,  12}},
		LeafPair<3> {Leaf(5), {12, 4,  12}},
		LeafPair<3> {Leaf(6), {4,  12, 12}},
		LeafPair<3> {Leaf(7), {12, 12, 12}},
	},
	// Deep octree with many leafs at the same point.
	std::vector<LeafPair<3> > {
		LeafPair<3> {Leaf(0), {13, 13, 13}},
		LeafPair<3> {Leaf(1), {13, 13, 13}},
		LeafPair<3> {Leaf(2), {13, 13, 13}},
		LeafPair<3> {Leaf(4), {13, 13, 13}},
	},
	// Complex quadtree with points in many various locations.
	std::vector<LeafPair<3> > {
		LeafPair<3> {Leaf(0),  {1,  2,  1}},
		LeafPair<3> {Leaf(1),  {6,  2,  1}},
		LeafPair<3> {Leaf(2),  {6,  6,  1}},
		LeafPair<3> {Leaf(3),  {3,  2,  1}},
		LeafPair<3> {Leaf(4),  {2,  6,  1}},
		LeafPair<3> {Leaf(5),  {14, 6,  1}},
		LeafPair<3> {Leaf(6),  {6,  14, 1}},
		LeafPair<3> {Leaf(7),  {6,  10, 1}},
		LeafPair<3> {Leaf(8),  {2,  10, 1}},
		LeafPair<3> {Leaf(9),  {2,  14, 1}},
		
		LeafPair<3> {Leaf(10), {10, 6,  1}},
		LeafPair<3> {Leaf(11), {10, 2,  1}},
		LeafPair<3> {Leaf(12), {9,  9,  1}},
		LeafPair<3> {Leaf(13), {15, 1,  1}},
		LeafPair<3> {Leaf(14), {13, 3,  1}},
		LeafPair<3> {Leaf(15), {15, 3,  1}},
		LeafPair<3> {Leaf(16), {13, 1,  1}},
		LeafPair<3> {Leaf(17), {11, 9,  1}},
		LeafPair<3> {Leaf(18), {9,  11, 1}},
		LeafPair<3> {Leaf(19), {11, 11, 1}},
		
		LeafPair<3> {Leaf(20), {15, 9,  1}},
		LeafPair<3> {Leaf(21), {15, 13, 1}},
		LeafPair<3> {Leaf(22), {15, 11, 1}},
		LeafPair<3> {Leaf(23), {15, 15, 1}},
		LeafPair<3> {Leaf(24), {13, 9,  1}},
		LeafPair<3> {Leaf(25), {13, 13, 1}},
		LeafPair<3> {Leaf(26), {11, 13, 1}},
		LeafPair<3> {Leaf(27), {9,  13, 1}},
		LeafPair<3> {Leaf(28), {11, 15, 1}},
		LeafPair<3> {Leaf(29), {9,  15, 1}},
	},
});

// A list of leaf positions at every possible position.
static auto const singleLeafPairData = bdata::make([]() {
	std::vector<LeafPair<3> > result;
	std::size_t index = 0;
	for (Scalar x = 0.0; x <= 16.0; x += 1.0) {
		for (Scalar y = 0.0; y <= 16.0; y += 1.0) {
			for (Scalar z = 0.0; z <= 16.0; z += 1.0) {
				result.push_back(LeafPair<3> {Leaf(index), {x, y, z}});
			}
		}
	}
	return result;
}());

static auto const singleLeafData = bdata::make([]() {
	std::vector<Leaf> result;
	for (int data = -1; data < 30; ++data) {
		result.push_back(Leaf(data));
	}
	return result; 
}());

// Takes an octree and a list of leaf-position pairs that should be contained
// within it. Checks the structure of the octree to make sure that the leafs
// are located at appropriate locations within the octree.
template<std::size_t Dim>
static CheckOctreeResult checkOctree(
		TestOctree<Dim> const& octree,
		std::vector<LeafPair<Dim> > allLeafPairs);

// Does the same thing as checkOctree, except does a BOOST_REQUIRE that the
// octree has a valid state.
static std::string checkOctreeString(CheckOctreeResult check);

// Constructs an empty octree, and then inserts a number of points into it.
BOOST_DATA_TEST_CASE(
		OctreeFillTest,
		octreeData * leafPairsData,
		emptyOctree,
		leafPairs) {
	// Insert the points in one at a time, and check that the octree is valid in
	// between each insertion.
	TestOctree<3> octree = emptyOctree;
	std::vector<LeafPair<3> > addedLeafPairs;
	for (auto it = leafPairs.begin(); it != leafPairs.end(); ++it) {
		addedLeafPairs.push_back(*it);
		octree.insert(*it);
		CheckOctreeResult check = checkOctree(octree, addedLeafPairs);
		BOOST_REQUIRE_MESSAGE(
			check == CheckOctreeResult::Success,
			"failed when adding leaf with data " +
			std::to_string(std::get<Leaf>(*it).data) + ": " +
			checkOctreeString(check));
	}
}
/*
// Adds a single point to an octree.
BOOST_DATA_TEST_CASE(
		OctreeInsertTest,
		octreeData * leafPairsData * singleLeafPairData,
		octree,
		leafPairs,
		leafPair) {
	for (auto it = leafPairs.begin(); it != leafPairs.end(); ++it) {
		octree.insert(*it);
	}
	
	CheckOctreeResult check = checkOctree(octree, leafPairs);
	BOOST_REQUIRE_MESSAGE(
		check == CheckOctreeResult::Success,
		"failed to construct valid octree: " +
		checkOctreeString(check));
	
	octree.insert(leafPair);
	leafPairs.push_back(leafPair);
	
	BOOST_REQUIRE_MESSAGE(
		check == CheckOctreeResult::Success,
		"failed when adding leaf to octree: " +
		checkOctreeString(check));
}

// Removes a single point from an octree.
BOOST_DATA_TEST_CASE(
		OctreeEraseTest,
		octreeData * leafPairsData * singleLeafPairData,
		octree,
		leafPairs,
		leaf) {
	for (auto it = leafPairs.begin(); it != leafPairs.end(); ++it) {
		octree.insert(*it);
	}
	
	CheckOctreeResult check = checkOctree(octree, leafPairs);
	BOOST_REQUIRE_MESSAGE(
		check == CheckOctreeResult::Success,
		"failed to construct valid octree: " +
		checkOctreeString(check));
	
	auto octreeIt = std::find(
		octree.leafs().begin(),
		octree.leafs().end(),
		leaf);
	auto leafPairsIt = std::find(
		leafPairs.begin(),
		leafPairs.end(),
		[=](LeafPair<3> pair) {
			return std::get<Leaf>(pair) == leaf;
		});
	
	BOOST_REQUIRE_MESSAGE(
		(octreeIt == octree.leafs().end()) == (leafPairsIt == leafPairs.end()),
		"failed when searching octree (leaf mismatch)");
	
	if (octreeIt != octree.leafs().end()) {
		octreeIt = octree.erase(octreeIt);
		leafPairsIt = leafPairs.erase(leafPairsIt);
	}
	
	BOOST_REQUIRE_MESSAGE(
		check == CheckOctreeResult::Success,
		"failed when removing leaf from octree: " +
		checkOctreeString(check));
}
*/
template<std::size_t Dim>
CheckOctreeResult checkOctree(
		TestOctree<Dim> const& octree,
		std::vector<LeafPair<Dim> > allLeafPairs) {
	// Create a stack storing the points that belong to the current node.
	std::vector<std::vector<LeafPair<Dim> > > leafPairStack;
	leafPairStack.push_back(allLeafPairs);
	
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
		std::vector<LeafPair<Dim> > leafPairs(leafPairStack.back());
		leafPairStack.pop_back();
		
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
				std::get<Leaf>(leafPair));
			if (leaf == node.leafs().end()) {
				return CheckOctreeResult::LeafMissing;
			}
			if (leaf.position() != std::get<Vector<Dim> >(leafPair)) {
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
				std::vector<LeafPair<Dim> > childLeafPairs;
				
				// Remove the child's leafs from leafPairs and put them into
				// childLeafPairs instead.
				auto lastChildLeaf = std::partition(
					leafPairs.begin(),
					leafPairs.end(),
					[child](LeafPair<Dim> leafPair) {
						auto begin = child.leafs().begin();
						auto end = child.leafs().end();
						return std::find(
							begin,
							end,
							std::get<Leaf>(leafPair)) != end;
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
				leafPairStack.push_back(childLeafPairs);
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
	if (!leafPairStack.empty()) {
		return CheckOctreeResult::ChildCountMismatch;
	}
	
	return CheckOctreeResult::Success;
}

std::string checkOctreeString(CheckOctreeResult check) {
	std::string error;
	switch (check) {
	case CheckOctreeResult::RootHasParent:
		error = "root node has parent";
		break;
	case CheckOctreeResult::LeafDuplicate:
		error = "node contains duplicate leafs";
		break;
	case CheckOctreeResult::LeafMissing:
		error = "node is missing leaf";
		break;
	case CheckOctreeResult::DepthIncorrect:
		error = "node has incorrect depth";
		break;
	case CheckOctreeResult::LeafPositionMismatch:
		error = "leaf at incorrect position";
		break;
	case CheckOctreeResult::LeafOutOfBounds:
		error = "leaf position not inside node boundary";
		break;
	case CheckOctreeResult::NodeOverCapacity:
		error = "node over max capacity";
		break;
	case CheckOctreeResult::NodeOverDepth:
		error = "node over max depth";
		break;
	case CheckOctreeResult::NodeUnderCapacity:
		error = "node's children are unnecessary";
		break;
	case CheckOctreeResult::ChildParentMismatch:
		error = "child's parent reference is incorrect";
		break;
	case CheckOctreeResult::LeafNotInChild:
		error = "node's leaf is not in children";
		break;
	case CheckOctreeResult::LeafNotInParent:
		error = "child node's leaf is not in parent";
		break;
	case CheckOctreeResult::ChildCountMismatch:
		error = "node had incorrect child count";
		break;
	default:
		error = "";
		break;
	}
	return error;
}

