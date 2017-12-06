#include "types.h"

// Computes the force on a leaf as a result of a certain leaf field.
force_t leaf_field_to_force(
		leaf_moment_t moment,
		leaf_field_t field,
		vector_t position) {
	force_t result = { moment.charge * field.field };
	return result;
}

// Computes the force on a leaf as a result of a certain node field.
force_t node_field_to_force(
		leaf_moment_t moment,
		node_field_t field,
		vector_t position) {
	force_t result = { moment.charge * field.field };
	return result;
}

void kernel convert_leaf_fields_to_forces(
		index_t num_leafs,
		global leaf_t const* leafs,
		global index_t const* leaf_field_indices,
		global force_t* forces,
		index_t num_fields,
		global leaf_field_t const* fields) {
	
	index_t leaf_index = (index_t) get_global_id(0);
	if (leaf_index >= num_leafs) {
		return;
	}
	
	force_t net_force = { (vector_t) (0, 0, 0, 0) };
	
	leaf_moment_t moment = leafs[leaf_index].value.moment;
	vector_t position = leafs[leaf_index].position;
	
	index_t field_start = leaf_field_indices[leaf_index];
	index_t field_end = leaf_field_indices[leaf_index + 1];
	for (
			index_t field_index = field_start;
			field_index < field_end;
			++field_index) {
		leaf_field_t field = fields[field_index];
		force_t next_force = leaf_field_to_force(moment, field, position);
		net_force.force += next_force.force;
	}
	forces[leaf_index].force = net_force.force;
}

void kernel convert_node_fields_to_forces(
		index_t num_leafs,
		global leaf_t const* leafs,
		global index_t const* leaf_field_indices,
		global force_t* forces,
		index_t num_fields,
		global node_field_t const* fields) {
	
	index_t leaf_index = (index_t) get_global_id(0);
	if (leaf_index >= num_leafs) {
		return;
	}
	
	force_t net_force = { (vector_t) (0, 0, 0, 0) };
	
	leaf_moment_t moment = leafs[leaf_index].value.moment;
	vector_t position = leafs[leaf_index].position;
	
	index_t field_start = leaf_field_indices[leaf_index];
	index_t field_end = leaf_field_indices[leaf_index + 1];
	for (
			index_t field_index = field_start;
			field_index < field_end;
			++field_index) {
		node_field_t field = fields[field_index];
		force_t next_force = node_field_to_force(moment, field, position);
		net_force.force += next_force.force;
	}
	forces[leaf_index].force = net_force.force;
}

