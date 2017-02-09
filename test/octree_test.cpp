#define BOOST_TEST_MODULE OctreeTest

#include "octree.h"
#include "tensor.h"

#include <boost/test/unit_test.hpp>

using namespace nbody;

struct Leaf {
	int data;
	Leaf(int data) : data(data) {
	}
};

struct Node {
	int data;
	Node(int data = 0) : data(data) {
	}
};

BOOST_AUTO_TEST_CASE(CreateOctree) {
	Octree<Leaf, Node, 3> octree({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
	BOOST_CHECK_EQUAL(4, 2 * 2);
}

