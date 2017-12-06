#ifndef __NBODY_DEVICE_TYPES_H_
#define __NBODY_DEVICE_TYPES_H_

#ifdef __OPENCL_VERSION__
#define __KERNEL__
#elif defined __cplusplus
#undef __KERNEL__
#else
#error "Header was not used in either C++ or OpenCL code."
#endif

#ifndef __KERNEL__
#include <initializer_list>
#include "nbody/device/cl_includes.h"
#endif

// These typedefs are used to index a buffer that stores the sizes of the
// different types on the device.
#define VERIFY_LEAF_T_INDEX        (0)
#define VERIFY_NODE_T_INDEX        (1)
#define VERIFY_LEAF_VALUE_T_INDEX  (2)
#define VERIFY_NODE_VALUE_T_INDEX  (3)
#define VERIFY_LEAF_MOMENT_T_INDEX (4)
#define VERIFY_NODE_MOMENT_T_INDEX (5)
#define VERIFY_LEAF_FIELD_T_INDEX  (6)
#define VERIFY_NODE_FIELD_T_INDEX  (7)
#define VERIFY_INTERACTION_T_INDEX (8)
#define VERIFY_NUM_TYPES           (9)

#ifndef __KERNEL__
namespace nbody {
namespace device {
#endif


#ifdef __KERNEL__
typedef uint   index_t;
typedef int    index_diff_t;
typedef float  scalar_t;
typedef float4 vector_t;
typedef uchar  byte_t;
#endif

#ifndef __KERNEL__
typedef cl_uint  index_t;
typedef cl_int   index_diff_t;
typedef cl_float scalar_t;
typedef cl_uchar byte_t;

// This class mimics an array type (with the subscript operator) using the
// built-in OpenCL vector type. This type is important so that vectors will be
// aligned correctly.
class vector_t final {
	
private:
	cl_float4 _vec;
	
public:
	vector_t() : _vec() {
	}
	vector_t(std::initializer_list<scalar_t> list) : _vec() {
		auto it = list.begin();
		for (unsigned int index = 0; index < list.size(); ++index) {
			operator[](index) = *(it++);
		}
	}
	
	scalar_t& operator[](unsigned index) {
		return reinterpret_cast<cl_float*>(&_vec)[index];
	}
	scalar_t const& operator[](unsigned index) const {
		return reinterpret_cast<cl_float const*>(&_vec)[index];
	}
};
#endif


// Stores the set of moments of a leaf.
typedef struct {
	
	scalar_t charge;
	
} leaf_moment_t;


// Stores the set of moments of a node.
typedef struct {
	
	scalar_t charge;
	vector_t dipole_moment;
	vector_t quadrupole_cross_terms;
	vector_t quadrupole_trace_terms;
	
} node_moment_t;


// The set of data stored at each leaf.
typedef struct {
	
	vector_t velocity;
	scalar_t mass;
	leaf_moment_t moment;
	
} leaf_value_t;


// The set of data stored at each node.
typedef struct {
	
	node_moment_t moment;
	
} node_value_t;


// OpenCL-compatible version of Octree::LeafInternal.
typedef struct {
	
	vector_t position;
	leaf_value_t value;
	
} leaf_t;


// Open-CL compatible version of Octree::NodeInternal.
typedef struct {
	
	vector_t position;
	vector_t dimensions;
	
	index_t depth;
	index_t child_indices[9];
	index_diff_t parent_index;
	index_t sibling_index;
	index_t leaf_count;
	index_t leaf_index;
	
	byte_t has_children;
	
	node_value_t value;
	
} node_t;


// Stores information about an interaction between two different nodes.
typedef struct {
	
	// The first node.
	index_t node_a_index;
	// The second node.
	index_t node_b_index;
	// Each interaction involving the first node will have this unique index.
	// The exact interactions have different indices from the approximate ones.
	index_t node_a_interaction_index;
	// Similarly with the second node.
	index_t node_b_interaction_index;
	// Are the nodes enough apart that their interaction can be approximated?
	byte_t can_approx;
	// Can the interaction be reduced into a set of simpler interactions?
	byte_t can_reduce;
	
} interaction_t;


// Stores several terms in the Taylor series expansion of a field from a leaf.
// Because there's only one term in the series (the constant term), the point
// at which the expansion is made doesn't matter, so for now it is not included.
typedef struct {
	
	vector_t field;
	
} leaf_field_t;


// Stores several terms in the Taylor series expansion of a field from a node.
typedef struct {
	
	// The point about which the expansion is being made.
	vector_t point;
	vector_t field;
	
} node_field_t;

// Stores a force acting on a leaf. Can include higher order forces such as
// torques and possibly even weird forces that can act on quadrupoles.
typedef struct {
	
	vector_t force;
	
} force_t;


#ifndef __KERNEL__
}
}
#endif

#endif

