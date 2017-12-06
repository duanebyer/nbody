#include "types.h"

// The first step in computing the moments of a set of particles. All child-less
// nodes are identified, and the particles contained within them are used to
// calculate their moments.
void kernel compute_moments_from_leafs(
		index_t num_leafs,
		global leaf_t const* leafs,
		index_t num_nodes,
		global node_t* nodes,
		global index_t* processed_node_indices) {
	
	index_t node_index = (index_t) get_global_id(0);
	if (node_index >= num_nodes) {
		return;
	}
	node_t node = nodes[node_index];
	
	if (!node.has_children) {
		processed_node_indices[node_index] = node_index;
		
		scalar_t charge = 0.0;
		vector_t dipole_moment = 0.0;
		vector_t quadrupole_cross_terms = 0.0;
		vector_t quadrupole_trace_terms = 0.0;
		
		vector_t center = node.position + node.dimensions / (scalar_t) 2;
		
		// Sum contributions from each of the particles contained within the
		// node.
		index_t leaf_start = node.leaf_index;
		index_t leaf_end = node.leaf_index + node.leaf_count;
		for (
				index_t leaf_index = leaf_start;
				leaf_index < leaf_end;
				++leaf_index) {
			// Get the intrinsic moments of the leaf.
			leaf_moment_t leaf_moment = leafs[leaf_index].value.moment;
			vector_t r = leafs[leaf_index].position - center;
			scalar_t q = leaf_moment.charge;
			
			// Compute the contribution of the leaf the moments of the node.
			charge += q;
			dipole_moment += q * r;
			quadrupole_cross_terms += (scalar_t) 3 * q * (vector_t) (
				r.y * r.z,
				r.x * r.z,
				r.x * r.y,
				0);
			quadrupole_trace_terms += q * (vector_t) (
				(scalar_t) 2 * r.x * r.x - r.y * r.y - r.z * r.z,
				(scalar_t) 2 * r.y * r.y - r.x * r.x - r.z * r.z,
				(scalar_t) 2 * r.z * r.z - r.x * r.x - r.y * r.y,
				0);
		}
		
		nodes[node_index].value.moment.charge = charge;
		nodes[node_index].value.moment.dipole_moment = dipole_moment;
		nodes[node_index].value.moment.quadrupole_cross_terms =
			quadrupole_cross_terms;
		nodes[node_index].value.moment.quadrupole_trace_terms =
			quadrupole_trace_terms;
	}
	else {
		processed_node_indices[node_index] = 0;
	}
}

// The second step in computing the moments of a set of particles. Node are read
// from a list that identifies which nodes have valid moments calculated. These
// nodes can then be used to calculate moments for their parents.
void kernel compute_moments_from_nodes(
		index_t num_nodes,
		global node_t* nodes,
		index_t num_processed_nodes,
		global index_t const* processed_node_indices,
		global index_t* new_processed_node_indices,
		index_t num_nodes_to_scan) {
	
	index_t processed_start = (index_t) (num_nodes_to_scan * get_global_id(0));
	index_t processed_end = min(
		processed_start + num_nodes_to_scan,
		num_processed_nodes);
	if (processed_start >= num_processed_nodes) {
		return;
	}
	
	// Loop over a small range of the processed nodes, looking for the one that
	// is the first of its siblings (looping backwards to avoid data race).
	for (
			index_t processed_index = processed_start;
			processed_index < processed_end;
			++processed_index) {
		// Get the index of one of the processed nodes.
		index_t node_index = processed_node_indices[processed_index];
		node_t node = nodes[node_index];
		
		if (node.sibling_index != 0) {
			continue;
		}
		
		// Then iterate through the next 7 siblings afterwards to see if they
		// have all been processed as well.
		bool valid_node = true;
		index_t sibling_index = node_index;
		for (
				index_t sibling_num = 1;
				sibling_num < 8;
				++sibling_num) {
			sibling_index += nodes[sibling_index].child_indices[8];
			if (
					processed_node_indices[processed_index + sibling_num] !=
					sibling_index) {
				valid_node = false;
				break;
			}
		}
		
		// If it turns out that they are all in the processed nodes, then it's
		// time to calculate the moments of their parent, and then to add their
		// parent to the processed nodes while removing them from the processed
		// nodes.
		if (valid_node) {
			scalar_t charge = 0.0;
			vector_t dipole_moment = 0.0;
			vector_t quadrupole_cross_terms = 0.0;
			vector_t quadrupole_trace_terms = 0.0;
			
			for (index_t sibling_num = 0; sibling_num < 8; ++sibling_num) {
				index_t sibling_index =
					processed_node_indices[processed_index + sibling_num];
				node_moment_t node_moment = nodes[sibling_index].value.moment;
				
				charge += node_moment.charge;
				dipole_moment += node_moment.dipole_moment;
				quadrupole_cross_terms += node_moment.quadrupole_cross_terms;
				quadrupole_trace_terms += node_moment.quadrupole_trace_terms;
				
				// Remove child from the processed nodes.
				new_processed_node_indices[processed_index + sibling_num] = 0;
			}
			
			// Set parent's moments.
			index_t parent_index = node_index + node.parent_index;
			nodes[parent_index].value.moment.charge = charge;
			nodes[parent_index].value.moment.dipole_moment = dipole_moment;
			nodes[parent_index].value.moment.quadrupole_cross_terms =
				quadrupole_cross_terms;
			nodes[parent_index].value.moment.quadrupole_trace_terms =
				quadrupole_trace_terms;
			
			// Add parent to the processed nodes.
			new_processed_node_indices[processed_index] = parent_index;
			
			// Now that we've dealt with all of the siblings, we can move
			// the processed_index forward a few steps.
			processed_index += 8 - 1;
		}
	}
}

