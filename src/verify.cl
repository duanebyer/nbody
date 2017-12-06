#include "types.h"

void kernel verify_device_type_sizes(global uint* sizes) {
	sizes[VERIFY_LEAF_T_INDEX]        = sizeof(leaf_t);
	sizes[VERIFY_NODE_T_INDEX]        = sizeof(node_t);
	sizes[VERIFY_LEAF_VALUE_T_INDEX]  = sizeof(leaf_value_t);
	sizes[VERIFY_NODE_VALUE_T_INDEX]  = sizeof(node_value_t);
	sizes[VERIFY_LEAF_MOMENT_T_INDEX] = sizeof(leaf_moment_t);
	sizes[VERIFY_NODE_MOMENT_T_INDEX] = sizeof(node_moment_t);
	sizes[VERIFY_LEAF_FIELD_T_INDEX]  = sizeof(leaf_field_t);
	sizes[VERIFY_NODE_FIELD_T_INDEX]  = sizeof(node_field_t);
	sizes[VERIFY_INTERACTION_T_INDEX] = sizeof(interaction_t);
}

