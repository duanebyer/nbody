#include "types.h"

#ifndef PARTICLE_RADIUS
#define PARTICLE_RADIUS ((scalar_t) 0.01)
#endif

#ifndef FORCE_CONSTANT
#define FORCE_CONSTANT ((scalar_t) -1.0)
#endif

typedef struct {
	leaf_field_t field_a;
	leaf_field_t field_b;
} leaf_field_pair_t;

// Computes the field of two leafs on each other. Returns a pair.
leaf_field_pair_t leaf_moment_field(
		leaf_moment_t moment_a,
		leaf_moment_t moment_b,
		vector_t position_a,
		vector_t position_b) {
	vector_t r = position_b - position_a;
	scalar_t r_mag = sqrt(dot(r, r) + PARTICLE_RADIUS * PARTICLE_RADIUS);
	vector_t unscaled_field = FORCE_CONSTANT * r / (r_mag * r_mag * r_mag);
	leaf_field_pair_t result = {
		// Field on A (from B).
		{ -moment_b.charge * unscaled_field },
		// Field on B (from A).
		{ +moment_a.charge * unscaled_field }
	};
	return result;
}

// Computes the field of a node at a certain point.
node_field_t node_moment_field(
		node_moment_t source_moment,
		vector_t source_position,
		vector_t target_position) {
	vector_t r = target_position - source_position;
	scalar_t r_mag = sqrt(dot(r, r) + PARTICLE_RADIUS * PARTICLE_RADIUS);
	vector_t unscaled_field = FORCE_CONSTANT * r / (r_mag * r_mag * r_mag);
	node_field_t result = {
		target_position,
		source_moment.charge * unscaled_field
	};
	return result;
}

void kernel compute_leaf_interaction_fields(
		// Leafs of the octree.
		index_t num_leafs,
		global leaf_t const* leafs,
		// The index in the field array that each leaf maps to.
		global index_t const* leaf_field_indices,
		// Nodes of the octree.
		index_t num_nodes,
		global node_t const* nodes,
		// Number of leaf counts for each node interaction.
		global index_t const* node_max_interactions_leaf_count,
		// A set of leaf interactions to be computed.
		index_t num_interactions,
		global interaction_t const* interactions,
		// Array of all fields on particles.
		index_t num_fields,
		global leaf_field_t* fields) {
	
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
	
	// Must break both of the nodes down into their leafs and compute every
	// force between them.
	index_t lid_a = (index_t) get_local_id(0);
	index_t lid_b = (index_t) get_local_id(1);
	
	// Break down a subset of the pairs of leafs.
	index_t num_leafs_a_per =
		get_local_size(0) / node_a.leaf_count +
		(get_local_size(0) % node_a.leaf_count != 0);
	index_t num_leafs_b_per =
		get_local_size(1) / node_b.leaf_count +
		(get_local_size(1) % node_b.leaf_count != 0);
	
	// Loop through a rectangular subset of all possible pairs.
	index_t leaf_a_start = node_a.leaf_index + lid_a * num_leafs_a_per;
	index_t leaf_b_start = node_b.leaf_index + lid_b * num_leafs_b_per;
	index_t leaf_a_end = node_a.leaf_index +
		min((lid_a + 1) * num_leafs_a_per, node_a.leaf_count);
	index_t leaf_b_end = node_b.leaf_index +
		min((lid_b + 1) * num_leafs_b_per, node_b.leaf_count);
	
	for (
			index_t leaf_a_index = leaf_a_start;
			leaf_a_index < leaf_a_end;
			++leaf_a_index) {
		for (
				index_t leaf_b_index = leaf_b_start;
				leaf_b_index < leaf_b_end;
				++leaf_b_index) {
			if (
					interaction.node_a_index == interaction.node_b_index &&
					leaf_b_index >= leaf_a_index) {
				continue;
			}
			leaf_t leaf_a = leafs[leaf_a_index];
			leaf_t leaf_b = leafs[leaf_b_index];
			
			// Calculate the field of node a on node b.
			leaf_moment_t moment_a = leafs[leaf_a_index].value.moment;
			leaf_moment_t moment_b = leafs[leaf_b_index].value.moment;
			leaf_field_pair_t field_pair = leaf_moment_field(
				moment_a,
				moment_b,
				leaf_a.position,
				leaf_b.position);
			
			// Determine the index of the force in the array of forces. Each
			// leaf has a set of 'num_leafs_per_node' forces for each leaf in
			// the other node involved in the interaction (repeat for each
			// interaction).
			index_t interaction_a_offset =
				node_max_interactions_leaf_count[node_a_index] *
				interaction.node_a_interaction_index;
			index_t interaction_b_offset =
				node_max_interactions_leaf_count[node_b_index] *
				interaction.node_b_interaction_index;
			index_t field_a_index =
				leaf_field_indices[leaf_a_index] +
				interaction_a_offset +
				(leaf_b_index - node_b.leaf_index);
			index_t field_b_index =
				leaf_field_indices[leaf_b_index] +
				interaction_b_offset +
				(leaf_a_index - node_a.leaf_index);
			
			fields[field_a_index] = field_pair.field_a;
			fields[field_b_index] = field_pair.field_b;
		}
	}
}

// The second step in computing the field on a set of particles. This kernel
// only calculates node fields. Precise ones need to be calculated using the
// compute_leaf_interaction_fields.
void kernel compute_node_interaction_fields(
		// The index in the field array that each leaf maps to.
		index_t num_leafs,
		global index_t const* node_field_indices,
		// Nodes of the octree.
		index_t num_nodes,
		global node_t const* nodes,
		// Cumulative number of interactions each node is involved in.
		global index_t const* node_num_parent_interactions,
		// A set of node interactions to be computed.
		index_t num_interactions,
		global interaction_t const* interactions,
		// Array of all fields on particles.
		index_t num_fields,
		global node_field_t* fields) {
	
	index_t interaction_index = (index_t) (get_group_id(0) / 2);
	bool use_node_a = (bool) (get_group_id(0) % 2);
	if (interaction_index >= num_interactions) {
		return;
	}
	interaction_t interaction = interactions[interaction_index];
	index_t target_node_index = use_node_a ?
		interaction.node_a_index : interaction.node_b_index;
	index_t source_node_index = use_node_a ?
		interaction.node_b_index : interaction.node_a_index;
	
	index_t target_node_interaction_index = use_node_a ?
		interaction.node_a_interaction_index :
		interaction.node_b_interaction_index;
	
	node_t target_node = nodes[target_node_index];
	node_t source_node = nodes[source_node_index];
	
	// Calculate the field of the source node.
	node_moment_t source_moment = nodes[source_node_index].value.moment;
	node_field_t field = node_moment_field(
		source_moment,
		source_node.position + source_node.dimensions / 2,
		target_node.position + target_node.dimensions / 2);
	
	index_t lid = (index_t) get_local_id(0);
	
	index_t num_leafs_per_item = get_local_size(0) / target_node.leaf_count +
		(get_local_size(0) % target_node.leaf_count != 0);
	
	index_t leaf_start = target_node.leaf_index + lid * num_leafs_per_item;
	index_t leaf_end = target_node.leaf_index +
		min((lid + 1) * num_leafs_per_item, target_node.leaf_count);
	
	// Apply the field of the source node to every leaf in the target node.
	for (index_t leaf_index = leaf_start; leaf_index < leaf_end; ++leaf_index) {
		index_t field_index =
			node_field_indices[leaf_index] +
			node_num_parent_interactions[target_node_index] +
			target_node_interaction_index;
		fields[field_index] = field;
	}
}

