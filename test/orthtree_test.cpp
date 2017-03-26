#define BOOST_TEST_MODULE OrthtreeTest

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "orthtree.h"

namespace bdata = boost::unit_test::data;
using namespace nbody;

struct LeafValue;
struct NodeValue;

using Scalar = double;
using Point = std::array<Scalar, 3>;
using LeafPair = std::tuple<LeafValue, Point>;
using Octree = Orthtree<3, Point, LeafValue, NodeValue>;

struct LeafValue {
	std::size_t data;
	explicit LeafValue(int data) : data(data) {
	}
	bool operator==(LeafValue const& other) const {
		return data == other.data;
	}
	bool operator!=(LeafValue const& other) const {
		return data != other.data;
	}
};

struct NodeValue {
	std::size_t data;
	explicit NodeValue(int data = 0) : data(data) {
	}
	bool operator==(NodeValue const& other) const {
		return data == other.data;
	}
	bool operator!=(NodeValue const& other) const {
		return data != other.data;
	}
};

enum class CheckOrthtreeResult {
	Success,
	RootHasParent,
	LeafExtra,
	LeafMissing,
	DepthIncorrect,
	LeafOutOfBounds,
	NodeOverCapacity,
	NodeOverDepth,
	NodeUnderCapacity,
	ChildParentMismatch,
	LeafNotInChild,
	LeafNotInParent,
	ChildCountMismatch,
};

// Takes an orthtree and a list of leaf-position pairs that should be contained
// within it. Checks the structure of the orthtree to make sure that the leafs
// are located at appropriate locations within the orthtree.
template<
	typename LeafPair,
	std::size_t Dim, typename Vector, typename LeafValue, typename NodeValue>
static CheckOrthtreeResult checkOrthtree(
		Orthtree<Dim, Vector, LeafValue, NodeValue> const& orthtree,
		std::vector<LeafPair> allLeafPairs);

std::string to_string(Point const& point);
std::string to_string(LeafValue const& leafValue);
std::string to_string(LeafPair const& pair);
std::string to_string(Octree const& octree);
std::string to_string(CheckOrthtreeResult check);

// This is needed so that Boost can print out test messages.
namespace boost {
namespace test_tools {
namespace tt_detail {

template<>
struct print_log_value<Point> {
	void operator()(std::ostream& os, Point const& point) {
		os << to_string(point);
	}
};
template<>
struct print_log_value<LeafValue> {
	void operator()(std::ostream& os, LeafValue const& leafValue) {
		os << to_string(leafValue);
	}
};
template<>
struct print_log_value<LeafPair> {
	void operator()(std::ostream& os, LeafPair const& pair) {
		os << to_string(pair);
	}
};
template<>
struct print_log_value<Octree> {
	void operator()(std::ostream& os, Octree const& octree) {
		os << to_string(octree);
	}
};
template<>
struct print_log_value<CheckOrthtreeResult> {
	void operator()(std::ostream& os, CheckOrthtreeResult check) {
		os << to_string(check);
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

}
}
}

// A collection of different octrees with various parameters.
static auto const octreeData =
	bdata::make(Octree({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 3, 4)) +
	bdata::make(Octree({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 3, 0)) +
	bdata::make(Octree({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 3, 1)) +
	bdata::make(Octree({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 3, 64)) +
	bdata::make(Octree({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 1, 64)) +
	bdata::make(Octree({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 64, 4)) +
	bdata::make(Octree({0.0, 0.0, 0.0}, {16.0, 16.0, 16.0}, 64, 0)) +
	bdata::make(Octree({-48.0, -32.0, 8.0}, {+64.0, +128.0, 4.0}, 3, 4));

// A set of leaf pair lists that can be used to construct octrees.
static auto const leafPairsData = bdata::make(
	std::vector<std::vector<LeafPair> > {
		// Shallow octree with a single point in each octant.
		std::vector<LeafPair> {
			LeafPair {LeafValue(0), {4,  4,  4}},
			LeafPair {LeafValue(1), {12, 4,  4}},
			LeafPair {LeafValue(2), {4,  12, 4}},
			LeafPair {LeafValue(3), {12, 12, 4}},
			LeafPair {LeafValue(4), {4,  4,  12}},
			LeafPair {LeafValue(5), {12, 4,  12}},
			LeafPair {LeafValue(6), {4,  12, 12}},
			LeafPair {LeafValue(7), {12, 12, 12}},
		},
		// Deep octree with many leafs at the same point.
		std::vector<LeafPair> {
			LeafPair {LeafValue(0), {13, 13, 13}},
			LeafPair {LeafValue(1), {13, 13, 13}},
			LeafPair {LeafValue(2), {13, 13, 13}},
			LeafPair {LeafValue(3), {13, 13, 13}},
		},
		// Complex quadtree with points in many various locations.
		std::vector<LeafPair> {
			LeafPair {LeafValue(0),  {1,  2,  1}},
			LeafPair {LeafValue(1),  {6,  2,  1}},
			LeafPair {LeafValue(2),  {6,  6,  1}},
			LeafPair {LeafValue(3),  {3,  2,  1}},
			LeafPair {LeafValue(4),  {2,  6,  1}},
			LeafPair {LeafValue(5),  {14, 6,  1}},
			LeafPair {LeafValue(6),  {6,  14, 1}},
			LeafPair {LeafValue(7),  {6,  10, 1}},
			LeafPair {LeafValue(8),  {2,  10, 1}},
			LeafPair {LeafValue(9),  {2,  14, 1}},
			
			LeafPair {LeafValue(10), {10, 6,  1}},
			LeafPair {LeafValue(11), {10, 2,  1}},
			LeafPair {LeafValue(12), {9,  9,  1}},
			LeafPair {LeafValue(13), {15, 1,  1}},
			LeafPair {LeafValue(14), {13, 3,  1}},
			LeafPair {LeafValue(15), {15, 3,  1}},
			LeafPair {LeafValue(16), {13, 1,  1}},
			LeafPair {LeafValue(17), {11, 9,  1}},
			LeafPair {LeafValue(18), {9,  11, 1}},
			LeafPair {LeafValue(19), {11, 11, 1}},
			
			LeafPair {LeafValue(20), {15, 9,  1}},
			LeafPair {LeafValue(21), {15, 13, 1}},
			LeafPair {LeafValue(22), {15, 11, 1}},
			LeafPair {LeafValue(23), {15, 15, 1}},
			LeafPair {LeafValue(24), {13, 9,  1}},
			LeafPair {LeafValue(25), {13, 13, 1}},
			LeafPair {LeafValue(26), {11, 13, 1}},
			LeafPair {LeafValue(27), {9,  13, 1}},
			LeafPair {LeafValue(28), {11, 15, 1}},
			LeafPair {LeafValue(29), {9,  15, 1}},
		},
	}
);

// A list of leaf positions at every possible position.
static auto const singleLeafPairData = bdata::make([]() {
	std::vector<LeafPair> result;
	std::size_t index = 0;
	for (Scalar x = 0.0; x <= 16.0; x += 1.0) {
		for (Scalar y = 0.0; y <= 16.0; y += 1.0) {
			for (Scalar z = 0.0; z <= 16.0; z += 1.0) {
				result.push_back(LeafPair {LeafValue(index), {x, y, z}});
			}
		}
	}
	return result;
}());

static auto const singleLeafData = bdata::make([]() {
	std::vector<LeafValue> result;
	for (int data = -1; data < 30; ++data) {
		result.push_back(LeafValue(data));
	}
	return result; 
}());

// Constructs an empty orthtree, and then inserts a number of points into it.
BOOST_DATA_TEST_CASE(
		OrthtreeFillTest,
		octreeData * leafPairsData,
		emptyOctree,
		leafPairs) {
	// Insert the points in one at a time, and check that the orthtree is valid in
	// between each insertion.
	Octree octree = emptyOctree;
	std::vector<LeafPair> addedLeafPairs;
	for (auto it = leafPairs.begin(); it != leafPairs.end(); ++it) {
		BOOST_TEST_CHECKPOINT("preparing to add " + to_string(*it));
		addedLeafPairs.push_back(*it);
		octree.insert(*it);
		BOOST_TEST_CHECKPOINT("finished adding " + to_string(*it));
		CheckOrthtreeResult check = checkOrthtree(octree, addedLeafPairs);
		BOOST_REQUIRE_EQUAL(check, CheckOrthtreeResult::Success);
	}
}

template<
	typename LeafPair,
	std::size_t Dim, typename Vector, typename LeafValue, typename NodeValue>
bool compareLeafPair(
		LeafPair pair,
		typename Orthtree<Dim, Vector, LeafValue, NodeValue>::
		ConstLeafReferenceProxy leaf) {
	return
		leaf.position == std::get<Point>(pair) &&
		leaf.value == std::get<LeafValue>(pair);
}

template<
	typename LeafPair,
	std::size_t Dim, typename Vector, typename LeafValue, typename NodeValue>
CheckOrthtreeResult checkOrthtree(
		Orthtree<Dim, Vector, LeafValue, NodeValue> const& orthtree,
		std::vector<LeafPair> allLeafPairs) {
	// Create a stack storing the points that belong to the current node.
	std::vector<std::vector<LeafPair> > leafPairsStack;
	leafPairsStack.push_back(allLeafPairs);
	
	// Check that the root node has no parent.
	if (orthtree.root()->hasParent) {
		return CheckOrthtreeResult::RootHasParent;
	}
	
	// Loop through all of the nodes.
	for (
			auto node = orthtree.root();
			node != orthtree.nodes().end();
			++node) {
		// Check that the current node has depth of +1 from its parent.
		if (node->depth != (node->hasParent ? node->parent->depth + 1 : 0)) {
			return CheckOrthtreeResult::DepthIncorrect;
		}
		
		// Take the top of the stack, and check whether each of the
		// leaf-position pairs are within the dimensions.
		std::vector<LeafPair> leafPairs(leafPairsStack.back());
		leafPairsStack.pop_back();
		
		if (leafPairs.size() > node->leafs.size()) {
			return CheckOrthtreeResult::LeafMissing;
		}
		if (leafPairs.size() < node->leafs.size()) {
			return CheckOrthtreeResult::LeafExtra;
		}
		
		for (auto leafPair : leafPairs) {
			// First, find the leaf within the orthtree.
			auto leaf = std::find_if(
				node->leafs.begin(),
				node->leafs.end(),
				std::bind(
					compareLeafPair<
						LeafPair,
						Dim, Vector, LeafValue, NodeValue>,
					leafPair,
					std::placeholders::_1));
			if (leaf == node->leafs.end()) {
				return CheckOrthtreeResult::LeafMissing;
			}
			// Then, make sure that it is contained within the bounds of the
			// node.
			for (std::size_t dim = 0; dim < Dim; ++dim) {
				Scalar position = node->position[dim];
				Scalar dimensions = node->dimensions[dim];
				if (!(
						leaf->position[dim] >= position &&
						leaf->position[dim] - position < dimensions)) {
					return CheckOrthtreeResult::LeafOutOfBounds;
				}
			}
		}
		
		// Next, check that the node's children contain all of its leafs. All of
		// of the node's leafs should belong to one and only one child.
		if (!node->hasChildren) {
			int depthSign =
				(node->depth > orthtree.maxDepth()) -
				(node->depth < orthtree.maxDepth());
			// If the node doesn't have children, then make sure that it doesn't
			// have too many leafs and that it isn't too deep.
			if (depthSign < 0 && node->leafs.size() > orthtree.nodeCapacity()) {
				return CheckOrthtreeResult::NodeOverCapacity;
			}
			if (depthSign > 0) {
				return CheckOrthtreeResult::NodeOverDepth;
			}
		}
		else {
			// Otherwise, make sure it doens't have too few leafs either.
			if (node->leafs.size() <= orthtree.nodeCapacity()) {
				return CheckOrthtreeResult::NodeUnderCapacity;
			}
			// Iterate over every child, and add its leafs to the stack (in
			// reverse order so that the children are added to the stack in
			// order).
			for (std::size_t childIndex = (1 << Dim); childIndex-- > 0; ) {
				auto child = node->children[childIndex];
				
				// Check that the child's parent is this node.
				if (child->parent != node) {
					return CheckOrthtreeResult::ChildParentMismatch;
				}
				// Create a vector to store the leaf-position pairs that belong
				// to the child.
				std::vector<LeafPair> childLeafPairs;
				
				// Remove the child's leafs from leafPairs and put them into
				// childLeafPairs instead.
				auto lastChildLeaf = std::partition(
					leafPairs.begin(),
					leafPairs.end(),
					[child](LeafPair leafPair) {
						return child->leafs.end() != std::find_if(
							child->leafs.begin(),
							child->leafs.end(),
							std::bind(
								compareLeafPair<
									LeafPair,
									Dim, Vector, LeafValue, NodeValue>,
								leafPair,
								std::placeholders::_1));
					});
				std::copy(
					leafPairs.begin(),
					lastChildLeaf,
					std::back_inserter(childLeafPairs));
				leafPairs.erase(leafPairs.begin(), lastChildLeaf);
				
				// Put the child leaf pairs onto the stack.
				if (childLeafPairs.size() != child->leafs.size()) {
					return CheckOrthtreeResult::LeafNotInParent;
				}
				leafPairsStack.push_back(childLeafPairs);
			}
			// Check that each of the leaf-position pairs belonged to at least
			// one of the children.
			if (!leafPairs.empty()) {
				return CheckOrthtreeResult::LeafNotInChild;
			}
		}
	}
	
	// The stack should be empty, except if one of the nodes didn't have the
	// right number of children.
	if (!leafPairsStack.empty()) {
		return CheckOrthtreeResult::ChildCountMismatch;
	}
	
	return CheckOrthtreeResult::Success;
}

std::string to_string(Point const& point) {
	std::ostringstream os;
	os << "<" << point[0];
	for (std::size_t dim = 1; dim < point.size(); ++dim) {
		os << ", " << point[dim];
	}
	os << ">";
	return os.str();
}

std::string to_string(LeafValue const& leaf) {
	std::ostringstream os;
	os << "LeafValue(" << leaf.data << ")";
	return os.str();
}

std::string to_string(LeafPair const& pair) {
	std::ostringstream os;
	os << "(";
	os << to_string(std::get<LeafValue>(pair)) << ", ";
	os << to_string(std::get<Point>(pair)) << ")";
	return os.str();
}

std::string to_string(Octree const& octree) {
	std::ostringstream os;
	os << "Octree(";
	os << "node capacity: " << octree.nodeCapacity() << ", ";
	os << "max depth: " << octree.maxDepth() << ", ";
	os << "position: ";
	os << to_string(octree.root()->position);
	os << ", ";
	os << "dimensions: ";
	os << to_string(octree.root()->dimensions);
	os << ")";
	return os.str();
}

std::string to_string(CheckOrthtreeResult check) {
	std::string error;
	switch (check) {
	case CheckOrthtreeResult::Success:
		error = "success";
		break;
	case CheckOrthtreeResult::RootHasParent:
		error = "root node has parent";
		break;
	case CheckOrthtreeResult::LeafExtra:
		error = "node contains extra leafs";
		break;
	case CheckOrthtreeResult::LeafMissing:
		error = "node is missing leaf";
		break;
	case CheckOrthtreeResult::DepthIncorrect:
		error = "node has incorrect depth";
		break;
	case CheckOrthtreeResult::LeafOutOfBounds:
		error = "leaf position not inside node boundary";
		break;
	case CheckOrthtreeResult::NodeOverCapacity:
		error = "node over max capacity";
		break;
	case CheckOrthtreeResult::NodeOverDepth:
		error = "node over max depth";
		break;
	case CheckOrthtreeResult::NodeUnderCapacity:
		error = "node's children are unnecessary";
		break;
	case CheckOrthtreeResult::ChildParentMismatch:
		error = "child's parent reference is incorrect";
		break;
	case CheckOrthtreeResult::LeafNotInChild:
		error = "node's leaf is not in children";
		break;
	case CheckOrthtreeResult::LeafNotInParent:
		error = "child node's leaf is not in parent";
		break;
	case CheckOrthtreeResult::ChildCountMismatch:
		error = "node had incorrect child count";
		break;
	default:
		error = "";
		break;
	}
	return error;
}

