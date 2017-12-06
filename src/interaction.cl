#include "types.h"

#ifndef NODE_APPROX_RATIO
#define NODE_APPROX_RATIO ((scalar_t) 0.5)
#endif

// The first step in computing the forces on a set of particles. A set of
// reducible interactions are reduced to create a list containing both reducable
// and irreducible interactions.
__attribute__((reqd_work_group_size(8, 8, 1)))
void kernel find_interactions(
		// The nodes that make up the octree.
		index_t num_nodes,
		global node_t const* nodes,
		// A set of reducible interactions.
		index_t num_interactions,
		global interaction_t const* interactions,
		// The results of reducing the above set of interactions. Size of this
		// array is 64 * num_interactions.
		global interaction_t* new_interactions) {
	
	index_t interaction_index = (index_t) get_group_id(0);
	if (interaction_index >= num_interactions) {
		return;
	}
	interaction_t interaction = interactions[interaction_index];
	
	// Get both nodes involved in the interaction.
	index_t node_a_index = interaction.node_a_index;
	index_t node_b_index = interaction.node_b_index;
	node_t node_a = nodes[node_a_index];
	node_t node_b = nodes[node_b_index];
	
	// Must break both of the nodes down into their children and go through
	// every possible interaction between them.
	index_t lid_a = (index_t) get_local_id(0);
	index_t lid_b = (index_t) get_local_id(1);
	
	// Find the indices of the two children that are going to partake in the
	// interaction. In the case that one of the nodes doesn't have children,
	// only consider the other node's children.
	index_t child_a_index = node_a_index;
	index_t child_b_index = node_b_index;
	if (node_a.has_children) {
		child_a_index += node_a.child_indices[lid_a];
	}
	if (node_b.has_children) {
		child_b_index += node_b.child_indices[lid_b];
	}
	node_t child_a = nodes[child_a_index];
	node_t child_b = nodes[child_b_index];
	
	// There are several special cases which should result in no interaction:
	if (
			child_a.leaf_count == 0 ||
			child_b.leaf_count == 0 ||
			(node_a_index == node_b_index && lid_b > lid_a) ||
			(!node_a.has_children && lid_a != 0) ||
			(!node_b.has_children && lid_b != 0)) {
		return;
	}
	
	// Calculate the distance between the centers of the two children.
	vector_t displacement =
		(child_b.position + child_b.dimensions / 2) -
		(child_a.position + child_a.dimensions / 2);
	scalar_t distance_sq = dot(displacement, displacement);
	
	// Calculate the maximum possible perpendicular average extent of the
	// children (from opposite corners of a node's volume).
	scalar_t extent_sum = child_a.dimensions.x + child_b.dimensions.x;
	scalar_t extent_sq = (scalar_t) (3.0 / 4.0) * extent_sum * extent_sum;
	
	// If the ratio of the extent to the distances is small enough, then
	// long distance approximations can be used.
	scalar_t approx_ratio_sq = NODE_APPROX_RATIO * NODE_APPROX_RATIO;
	bool can_approx =
		(child_a_index != child_b_index) &&
		(extent_sq / distance_sq < approx_ratio_sq);
	bool can_reduce =
		!can_approx &&
		(child_a.has_children || child_b.has_children);
	
	// Fill out the details of the new interaction.
	interaction_t new_interaction = {
		child_a_index,
		child_b_index,
		0,
		0,
		can_approx,
		can_reduce
	};
	
	// In the returning array, any interactions between node '0' and node '0'
	// will be removed, and the other interactions will be sorted into the
	// irreducible and the reducible interactions. The reducable interactions
	// will be substituted back into this kernel until there are none left.
	index_t new_interaction_index = 64 * interaction_index + 8 * lid_b + lid_a;
	new_interactions[new_interaction_index] = new_interaction;
}

// Fills out the interactions so that they know what index they are within the
// set of all interactions acting on a node.
void kernel compute_interaction_indices(
		index_t num_nodes,
		// A record of how many interactions each node is part of.
		global index_t* node_num_interactions,
		// The interactions.
		index_t num_interactions,
		global interaction_t* interactions) {
	
	index_t interaction_index = (index_t) get_global_id(0);
	if (interaction_index >= num_interactions) {
		return;
	}
	interaction_t interaction = interactions[interaction_index];
	
	// Increment the number of interactions that the nodes are a part of.
	interaction.node_a_interaction_index =
		atomic_inc(node_num_interactions + interaction.node_a_index);
	interaction.node_b_interaction_index =
		atomic_inc(node_num_interactions + interaction.node_b_index);
}

// Determine the maximum number of leafs acting on each other in a leaf
// interaction.
void kernel compute_node_max_interactions_leaf_count(
		index_t num_nodes,
		global node_t const* nodes,
		// A record of the maximum number of leaf counts in any node that a node
		// has a leaf interaction with.
		global index_t* node_max_interactions_leaf_count,
		// The interactions.
		index_t num_interactions,
		global interaction_t const* interactions) {
	
	index_t interaction_index = (index_t) get_global_id(0);
	if (interaction_index >= num_interactions) {
		return;
	}
	interaction_t interaction = interactions[interaction_index];
	
	// Store the maximum number of 
	index_t node_a_index = interaction.node_a_index;
	index_t node_b_index = interaction.node_b_index;
	node_t node_a = nodes[node_a_index];
	node_t node_b = nodes[node_b_index];
	atomic_max(
		node_max_interactions_leaf_count + node_a_index,
		node_b.leaf_count);
	atomic_max(
		node_max_interactions_leaf_count + node_b_index,
		node_a.leaf_count);
}

