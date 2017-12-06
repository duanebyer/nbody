#include "nbody/open_cl_simulation.h"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iterator>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace nbody;

OpenClSimulation::OpenClSimulation(
		device::vector_t bounds,
		std::vector<Particle> particles,
		Scalar timeStep,
		std::ostream& log) :
		_octree(device::vector_t(), bounds),
		_time(0.0),
		_timeStep(timeStep),
		_log(log) {
	// Fill vectors with all of the leaf data.
	std::vector<device::leaf_value_t> leafValues;
	std::vector<device::vector_t> leafPositions;
	leafValues.reserve(particles.size());
	leafPositions.reserve(particles.size());
	for (std::size_t index = 0; index < particles.size(); ++index) {
		Particle particle = particles[index];
		device::leaf_value_t leafValue = {
			particle.velocity,
			particle.mass,
			particle.charge
		};
		leafValues.push_back(leafValue);
		leafPositions.push_back(particle.position);
	}
	
	// It's most efficient to add every particle at once.
	// FIXME: The node capacity is arbitrarily set at 8. Should be
	// adjustable by the user of this class.
	_octree = Octree(
		device::vector_t(), bounds,
		leafValues.begin(), leafValues.end(),
		leafPositions.begin(), leafPositions.end(),
		8);
	
	// Initialize OpenCL.
	initialize();
}

std::vector<OpenClSimulation::Particle> OpenClSimulation::particles() const {
	std::vector<Particle> result;
	result.reserve(_octree.leafs().size());
	for (
			Octree::ConstLeafIterator leafIt = _octree.cleafs().begin();
			leafIt != _octree.cleafs().end();
			++leafIt) {
		result.push_back({
			leafIt->position,
			leafIt->value.velocity,
			leafIt->value.mass,
			leafIt->value.moment.charge
		});
	}
	return result;
}

OpenClSimulation::Scalar OpenClSimulation::step() {
	_log << "Starting a new step (t=" << _time << ").\n";
	
	// Octree buffers.
	_log << "Computing moments.\n";
	OctreeBuffers octreeBuffers = computeOctreeBuffers();
	
	// A set of interactions that still need to be processed (starting with just
	// the root node interacting with itself).
	UnprocessedInteractionBuffers unprocessedInteractions;
	IntegrationBuffers integrationBuffers;
	do {
		// Interactions.
		_log << "Computing interactions.\n";
		InteractionBuffers interactionBuffers = computeInteractionBuffers(
			octreeBuffers,
			unprocessedInteractions);
		// Fields and forces.
		_log << "Computing forces.\n";
		ForceBuffers forceBuffers = computeForceBuffers(
			octreeBuffers,
			interactionBuffers);
		// Integration.
		_log << "Computing integration.\n";
		integrationBuffers = computeIntegrationBuffers(
			forceBuffers,
			integrationBuffers);
	}
	while (!unprocessedInteractions.finished());
	
	_log << "Updating octree.\n";
	updateOctree(integrationBuffers);
	
	_time += _timeStep;
	_log << "Step finished.\n";
	return _time;
}

void OpenClSimulation::updateOctree(IntegrationBuffers integrationBuffers) {
	// Update the leaf velocities.
	for (
			Octree::LeafListSizeType leafIndex = 0;
			leafIndex < _octree.leafs().size();
			++leafIndex) {
		Octree::LeafIterator leafIt = _octree.leafs().begin() + leafIndex;
		device::vector_t velocity = integrationBuffers.newVelocities[leafIndex];
		leafIt->value.velocity = velocity;
	}
	
	// Move all of the leafs to their new positions.
	_octree.move(
		_octree.leafs().begin(), _octree.leafs().end(),
		integrationBuffers.newPositions.begin(),
		integrationBuffers.newPositions.end());
}

OpenClSimulation::OctreeBuffers OpenClSimulation::computeOctreeBuffers() {
	// First, create buffers to hold the leafs and the nodes.
	device::BufferWrapper<device::leaf_t> leafs = createBuffer(
		device::IOFlag::Read,
		_octree.leafs().size(),
		reinterpret_cast<device::leaf_t const*>(_octree.leafs().data()));
	device::BufferWrapper<device::node_t> nodes = createBuffer(
		device::IOFlag::ReadWrite,
		_octree.nodes().size(),
		reinterpret_cast<device::node_t const*>(_octree.nodes().data()));
	// Create additional buffers to store the processed nodes and the updated
	// processed nodes.
	device::BufferWrapper<device::index_t> processedNodes =
		createBuffer<device::index_t>(
			device::IOFlag::Read, _octree.nodes().size());
	device::BufferWrapper<device::index_t> newProcessedNodes =
		createBuffer<device::index_t>(
			device::IOFlag::Write, _octree.nodes().size());
	
	// Do the first pass: calculate the moments of the child-less nodes.
	kernelComputeMomentsFromLeafs(leafs, nodes, newProcessedNodes);
	
	// Now recursively move up the octree until all node moments have been
	// computed.
	Octree::LeafListSizeType numProcessedNodes = leafs.size();
	while (numProcessedNodes != 0) {
		// Remove zeros from the processed nodes and collapse the remaining
		// entries to the front of the array.
		device::index_t* processedNodesData = newProcessedNodes.map(
			device::IOFlag::ReadWrite);
		numProcessedNodes = 0;
		for (device::index_t i = 0; i < numProcessedNodes; ++i) {
			if (processedNodesData[i] != 0) {
				processedNodesData[numProcessedNodes] = processedNodesData[i];
				++numProcessedNodes;
			}
		}
		newProcessedNodes.unmap(processedNodesData);
		// Move the result to the other buffer.
		newProcessedNodes.resize(numProcessedNodes, false, true);
		processedNodes.resize(numProcessedNodes, false, true);
		processedNodes.copyFrom(newProcessedNodes);
		
		// Call the kernel to reduce any nodes.
		kernelComputeMomentsFromNodes(nodes, processedNodes, newProcessedNodes);
	}
	
	return { leafs, nodes };
}

OpenClSimulation::InteractionBuffers OpenClSimulation::computeInteractionBuffers(
		OctreeBuffers octreeBuffers,
		UnprocessedInteractionBuffers& unprocessed) {
	// Determine how many of the unprocessed interactions will be processed
	// during this step.
	std::size_t maxProcessed =
		_deviceMaxBufferSize / (8 * 8 * sizeof(device::interaction_t));
	std::size_t numProcessed = std::min<std::size_t>(
		unprocessed.interactions.size(),
		maxProcessed);
	device::interaction_t* processedData =
		unprocessed.interactions.data() +
		unprocessed.interactions.size() -
		numProcessed;
	
	// Create buffers to hold the interactions.
	device::BufferWrapper<device::interaction_t> interactions =
		createBuffer<device::interaction_t>(
			device::IOFlag::Read,
			numProcessed,
			processedData);
	device::BufferWrapper<device::interaction_t> newInteractions =
		createBuffer<device::interaction_t>(
			device::IOFlag::Write,
			8 * 8 * numProcessed);
	
	// Remove the interactions that will be processed from the unprocessed list.
	unprocessed.interactions.resize(
		unprocessed.interactions.size() - numProcessed);
	// But reserve space for the new interactions that will be added later.
	unprocessed.interactions.reserve(
		unprocessed.interactions.size() + 64 * numProcessed);
	
	// Create buffers to hold the leaf and node interactions.
	device::BufferWrapper<device::interaction_t> leafInteractions =
		createBuffer<device::interaction_t>(device::IOFlag::Read, 0);
	device::BufferWrapper<device::interaction_t> nodeInteractions =
		createBuffer<device::interaction_t>(device::IOFlag::Read, 0);
	// Create buffers to hold the node counts per interaction.
	device::BufferWrapper<device::index_t> nodeNumLeafInteractions =
		createBuffer<device::index_t>(
			device::IOFlag::ReadWrite,
			octreeBuffers.nodes.size());
	device::BufferWrapper<device::index_t> nodeNumNodeInteractions =
		createBuffer<device::index_t>(
			device::IOFlag::ReadWrite,
			octreeBuffers.nodes.size());
	// Create a buffer to hold the max leaf count of a node's interactions.
	device::BufferWrapper<device::index_t> nodeMaxInteractionsLeafCount =
		createBuffer<device::index_t>(
			device::IOFlag::ReadWrite,
			_octree.nodes().size());
	
	nodeNumLeafInteractions.zero();
	nodeNumNodeInteractions.zero();
	nodeMaxInteractionsLeafCount.zero();
	
	// So long as there is still an interaction to reduce, perform one more
	// reduction step.
	if (interactions.size() != 0) {
		// Call the kernel to reduce the current set of interactions.
		newInteractions.zero();
		kernelFindInteractions(
			octreeBuffers.nodes,
			interactions,
			newInteractions);
		device::interaction_t* newInteractionsData =
			newInteractions.map(device::IOFlag::Read);
		
		// Loop through the new interactions and divide them into three sets:
		// reducible, leaf, and node interactions.
		for (
				std::size_t index = 0;
				index < newInteractions.size();
				++index) {
			device::interaction_t newInteraction = newInteractionsData[index];
			if (
					newInteraction.node_a_index == 0 &&
					newInteraction.node_b_index == 0) {
				// In this case, there is no interaction (placeholder value).
			}
			else if (newInteraction.can_reduce) {
				unprocessed.interactions.push_back(newInteraction);
			}
			else if (!newInteraction.can_approx) {
				unprocessed.leafInteractions.push_back(newInteraction);
			}
			else if (newInteraction.can_approx) {
				unprocessed.nodeInteractions.push_back(newInteraction);
			}
		}
		newInteractions.unmap(newInteractionsData);
	}
	
	// Determine how many leaf/node interactions can be calculated without
	// running out of memory.
	std::size_t numLeafInteractions = 0;
	std::size_t numNodeInteractions = 0;
	std::size_t leafInteractionsMemUsage = 0;
	std::size_t nodeInteractionsMemUsage = 0;
	
	// It isn't too expensive (I think!) to measure the space needed directly.
	// Regardless, this could easily be replaced by a heuristic.
	
	// FIXME: These expressions are not quite correct because of the effects of
	// when some nodes have more leafs than their capacity. Hard to fix, so the
	// leaf sizes are arbitrarily rescaled by a factor of two when calculating
	// memory usage.
	
	// First calculate the leaf interactions space needed.
	while (numLeafInteractions < unprocessed.leafInteractions.size()) {
		device::interaction_t interaction = unprocessed.leafInteractions[
			unprocessed.leafInteractions.size() -
			numLeafInteractions - 1];
		Octree::NodeIterator nodeA =
			_octree.nodes().begin() + interaction.node_a_index;
		Octree::NodeIterator nodeB =
			_octree.nodes().begin() + interaction.node_b_index;
		std::size_t nextMemUsage =
			2 * (2 * nodeA->leafs.size()) * (2 * nodeB->leafs.size()) *
			sizeof(device::leaf_field_t);
		if (leafInteractionsMemUsage + nextMemUsage >= _deviceMaxBufferSize) {
			break;
		}
		else {
			leafInteractionsMemUsage += nextMemUsage;
			++numLeafInteractions;
		}
	}
	
	// Then calculate the node interactions space needed.
	while (numNodeInteractions < unprocessed.nodeInteractions.size()) {
		device::interaction_t interaction = unprocessed.nodeInteractions[
			unprocessed.nodeInteractions.size() -
			numNodeInteractions - 1];
		Octree::NodeIterator nodeA =
			_octree.nodes().begin() + interaction.node_a_index;
		Octree::NodeIterator nodeB =
			_octree.nodes().begin() + interaction.node_b_index;
		std::size_t nextMemUsage =
			(2 * nodeA->leafs.size() + 2 * nodeB->leafs.size()) *
			sizeof(device::node_field_t);
		if (nodeInteractionsMemUsage + nextMemUsage >= _deviceMaxBufferSize) {
			break;
		}
		else {
			nodeInteractionsMemUsage += nextMemUsage;
			++numNodeInteractions;
		}
	}
	
	// Transfer the maximum interactions that can be processed to some buffers.
	leafInteractions.resize(numLeafInteractions);
	nodeInteractions.resize(numNodeInteractions);
	leafInteractions.write(
		unprocessed.leafInteractions.data() +
		unprocessed.leafInteractions.size() -
		numLeafInteractions);
	nodeInteractions.write(
		unprocessed.nodeInteractions.data() +
		unprocessed.nodeInteractions.size() -
		numNodeInteractions);
	unprocessed.leafInteractions.resize(
		unprocessed.leafInteractions.size() -
		numLeafInteractions);
	unprocessed.nodeInteractions.resize(
		unprocessed.nodeInteractions.size() -
		numNodeInteractions);
	
	// Compute the interaction indices separately for leaf and node
	// interactions.
	kernelComputeInteractionIndices(
		octreeBuffers.nodes,
		leafInteractions,
		nodeNumLeafInteractions);
	kernelComputeInteractionIndices(
		octreeBuffers.nodes,
		nodeInteractions,
		nodeNumNodeInteractions);
	
	// Compute max leafs that a node can interact with (by leaf interactions).
	kernelComputeNodeMaxInteractionsLeafCount(
		octreeBuffers.nodes,
		leafInteractions,
		nodeMaxInteractionsLeafCount);
	
	return {
		leafInteractions,
		nodeInteractions,
		nodeNumLeafInteractions,
		nodeNumNodeInteractions,
		nodeMaxInteractionsLeafCount
	};
}

device::index_t OpenClSimulation::computeLeafFieldIndices(
		device::BufferWrapper<device::index_t> nodeNumLeafInteractions,
		device::BufferWrapper<device::index_t> nodeMaxInteractionsLeafCount,
		device::BufferWrapper<device::index_t> leafFieldIndices) {
	// Map the buffers so they can be operated on in host.
	device::index_t* nodeNumLeafInteractionsData =
		nodeNumLeafInteractions.map(device::IOFlag::Read); 
	device::index_t* nodeMaxInteractionsLeafCountData =
		nodeMaxInteractionsLeafCount.map(device::IOFlag::Read);
	device::index_t* leafFieldIndicesData =
		leafFieldIndices.map(device::IOFlag::ReadWrite);
	device::node_t const* nodes =
		reinterpret_cast<device::node_t const*>(_octree.nodes().data());
	
	// For each leaf, add up all of the interactions that it is part of and use
	// that to give it a unique index in the fields array.
	leafFieldIndicesData[0] = 0;
	for (
			device::index_t nodeIndex = 0;
			nodeIndex < _octree.nodes().size();
			++nodeIndex) {
		device::node_t node = nodes[nodeIndex];
		// Skip nodes with children.
		if (!node.has_children) {
			device::index_t numFields =
				nodeMaxInteractionsLeafCountData[nodeIndex] *
				nodeNumLeafInteractionsData[nodeIndex];
			for (
					device::index_t leafIndex = node.leaf_index;
					leafIndex < node.leaf_index + node.leaf_count;
					++leafIndex) {
				device::index_t previous = leafFieldIndicesData[leafIndex];
				leafFieldIndicesData[leafIndex + 1] = previous + numFields;
			}
		}
	}
	
	// Return the total number of fields needed.
	device::index_t numFields = leafFieldIndicesData[_octree.leafs().size()];
	
	// Unmap the buffers.
	nodeNumLeafInteractions.unmap(nodeNumLeafInteractionsData);
	nodeMaxInteractionsLeafCount.unmap(nodeMaxInteractionsLeafCountData);
	leafFieldIndices.unmap(leafFieldIndicesData);
	
	return numFields;
}

device::index_t OpenClSimulation::computeNodeFieldIndices(
		device::BufferWrapper<device::index_t> nodeNumNodeInteractions,
		device::BufferWrapper<device::index_t> nodeNumNodeParentInteractions,
		device::BufferWrapper<device::index_t> nodeFieldIndices) {
	// Map both the buffers so they can be operated on in host.
	device::index_t* nodeNumNodeInteractionsData =
		nodeNumNodeInteractions.map(device::IOFlag::Read); 
	device::index_t* nodeNumNodeParentInteractionsData =
		nodeNumNodeParentInteractions.map(device::IOFlag::ReadWrite);
	device::index_t* nodeFieldIndicesData =
		nodeFieldIndices.map(device::IOFlag::ReadWrite);
	device::node_t const* nodes =
		reinterpret_cast<device::node_t const*>(_octree.nodes().data());
	
	// First, determine how many node interactions each node's ancestors are a
	// part of. This is important, since a node's leafs are affected by both its
	// interactions and its parent's interactions.
	for (
			device::index_t nodeIndex = 0;
			nodeIndex < _octree.nodes().size();
			++nodeIndex) {
		device::node_t node = nodes[nodeIndex];
		if (node.has_children) {
			for (device::index_t childNum = 0; childNum < 8; ++childNum) {
				device::index_t childIndex =
					nodeIndex + node.child_indices[childNum];
				nodeNumNodeParentInteractionsData[childIndex] +=
					nodeNumNodeInteractionsData[nodeIndex] +
					nodeNumNodeParentInteractionsData[nodeIndex];
			}
		}
	}
	
	// For each leaf, add up all of the interactions that it is part of and use
	// that to give it a unique index in the fields array.
	nodeFieldIndicesData[0] = 0;
	for (
			device::index_t nodeIndex = 0;
			nodeIndex < _octree.nodes().size();
			++nodeIndex) {
		device::node_t node = nodes[nodeIndex];
		// Skip nodes with children.
		if (!node.has_children) {
			device::index_t numFields =
				nodeNumNodeInteractionsData[nodeIndex] +
				nodeNumNodeParentInteractionsData[nodeIndex];
			for (
					device::index_t leafIndex = node.leaf_index;
					leafIndex < node.leaf_index + node.leaf_count;
					++leafIndex) {
				device::index_t previous = nodeFieldIndicesData[leafIndex];
				nodeFieldIndicesData[leafIndex + 1] = previous + numFields;
			}
		}
	}
	
	// Return the total number of fields needed.
	device::index_t numFields = nodeFieldIndicesData[_octree.leafs().size()];
	
	// Unmap the buffers.
	nodeNumNodeInteractions.unmap(nodeNumNodeInteractionsData);
	nodeNumNodeParentInteractions.unmap(nodeNumNodeParentInteractionsData);
	nodeFieldIndices.unmap(nodeFieldIndicesData);
	
	return numFields;
}

OpenClSimulation::ForceBuffers OpenClSimulation::computeForceBuffers(
		OctreeBuffers octreeBuffers,
		InteractionBuffers interactionBuffers) {
	// First, get the leaf field indices so that every field can be assigned a
	// location in the field array.
	device::BufferWrapper<device::index_t> leafFieldIndices =
		createBuffer<device::index_t>(
			device::IOFlag::Read,
			octreeBuffers.leafs.size() + 1);
	device::BufferWrapper<device::index_t> nodeFieldIndices =
		createBuffer<device::index_t>(
			device::IOFlag::Read,
			octreeBuffers.leafs.size() + 1);
	device::BufferWrapper<device::index_t> nodeNumNodeParentInteractions =
		createBuffer<device::index_t>(
			device::IOFlag::ReadWrite,
			octreeBuffers.nodes.size());
	
	nodeNumNodeParentInteractions.zero();
	
	std::size_t numLeafFields = computeLeafFieldIndices(
		interactionBuffers.nodeNumLeafInteractions,
		interactionBuffers.nodeMaxInteractionsLeafCount,
		leafFieldIndices);
	std::size_t numNodeFields = computeNodeFieldIndices(
		interactionBuffers.nodeNumNodeInteractions,
		nodeNumNodeParentInteractions,
		nodeFieldIndices);
	
	// Prepare the buffers to hold the fields.
	device::BufferWrapper<device::leaf_field_t> leafFields =
		createBuffer<device::leaf_field_t>(
			device::IOFlag::ReadWrite, numLeafFields);
	device::BufferWrapper<device::node_field_t> nodeFields =
		createBuffer<device::node_field_t>(
			device::IOFlag::ReadWrite, numNodeFields);
	
	leafFields.zero();
	nodeFields.zero();
	
	// Prepare the buffers to hold the forces.
	device::BufferWrapper<device::force_t> leafForces =
		createBuffer<device::force_t>(
			device::IOFlag::ReadWrite,
			octreeBuffers.leafs.size());
	device::BufferWrapper<device::force_t> nodeForces =
		createBuffer<device::force_t>(
			device::IOFlag::ReadWrite,
			octreeBuffers.leafs.size());
	
	leafForces.zero();
	nodeForces.zero();
	
	// Calculate the fields.
	kernelComputeLeafInteractionFields(
		octreeBuffers.leafs,
		octreeBuffers.nodes,
		interactionBuffers.leafInteractions,
		leafFieldIndices,
		interactionBuffers.nodeMaxInteractionsLeafCount,
		leafFields);
	kernelComputeNodeInteractionFields(
		octreeBuffers.nodes,
		interactionBuffers.nodeInteractions,
		nodeFieldIndices,
		nodeNumNodeParentInteractions,
		nodeFields);
	
	// Calculate the forces.
	kernelConvertLeafFieldsToForces(
		octreeBuffers.leafs,
		leafFieldIndices,
		leafFields,
		leafForces);
	kernelConvertNodeFieldsToForces(
		octreeBuffers.leafs,
		nodeFieldIndices,
		nodeFields,
		nodeForces);
	
	return {
		leafForces,
		nodeForces
	};
}

OpenClSimulation::IntegrationBuffers OpenClSimulation::computeIntegrationBuffers(
		ForceBuffers forceBuffers,
		IntegrationBuffers integrationBuffers) {
	// Resize the integration buffers to make sure there will be enough space
	// to hold the results.
	integrationBuffers.newPositions.resize(_octree.leafs().size());
	integrationBuffers.newVelocities.resize(_octree.leafs().size());
	
	// Map both sets of forces for reading.
	device::force_t* leafForcesData =
		forceBuffers.leafForces.map(device::IOFlag::Read);
	device::force_t* nodeForcesData =
		forceBuffers.nodeForces.map(device::IOFlag::Read);
	
	// Loop through every leaf and calculate it's new position and velocity.
	for (
			Octree::LeafListSizeType leafIndex = 0;
			leafIndex < _octree.leafs().size();
			++leafIndex) {
		// Get the calculated forces for updating leaf position.
		device::force_t leafForce = leafForcesData[leafIndex];
		device::force_t nodeForce = nodeForcesData[leafIndex];
		// Get the current leaf position.
		Octree::LeafIterator leafIt = _octree.leafs().begin() + leafIndex;
		device::vector_t position = leafIt->position;
		device::vector_t velocity = leafIt->value.velocity;
		device::scalar_t mass = leafIt->value.mass;
		
		// Perform simple leapfrog integration to update velocities and find the
		// new positions.
		for (unsigned int i = 0; i < 3; ++i) {
			device::scalar_t force = leafForce.force[i] + nodeForce.force[i];
			device::scalar_t oldVelocity = velocity[i];
			velocity[i] += force / mass * _timeStep;
			position[i] += oldVelocity * _timeStep;
		}
		integrationBuffers.newPositions[leafIndex] = position;
		integrationBuffers.newVelocities[leafIndex] = velocity;
	}
	
	forceBuffers.leafForces.unmap(leafForcesData);
	forceBuffers.nodeForces.unmap(nodeForcesData);
	
	return integrationBuffers;
}

void OpenClSimulation::initialize() {
	// Initialize OpenCL.
	_log << "Initializing OpenCL.\n";
	
	// Choose the default platform and device.
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	_platform = platforms[0];
	
	std::vector<cl::Device> devices;
	_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (devices.size() == 0) {
		throw std::runtime_error("OpenCL not supported");
	}
	_device = devices[0];
	
	// Output what is being used.
	std::string platformName = _platform.getInfo<CL_PLATFORM_NAME>();
	std::string deviceName = _device.getInfo<CL_DEVICE_NAME>();
	_log << "Platorm: " << platformName << "\n";
	_log << "Device:  " << deviceName << "\n";
	
	// Create the context and command queue.
	_context = cl::Context(_device);
	_queue = cl::CommandQueue(_context);
	
	// Determine the maximum allowed buffer size. Make sure it's larger than
	// some arbitrary small minimum.
	_device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &_deviceMaxBufferSize);
	if (_deviceMaxBufferSize < 1024 * 1024) {
		throw std::runtime_error("Device max buffer size is too small (<1 Mb)");
	}
	
	// Load all of the OpenCL sources.
	cl::Program programVerify = buildSourceFile("verify.cl");
	cl::Program programMoment = buildSourceFile("moment.cl");
	cl::Program programInteraction = buildSourceFile("interaction.cl");
	cl::Program programField = buildSourceFile("field.cl");
	cl::Program programForce = buildSourceFile("force.cl");
	
	// Get the kernels.
	_kernelVerifyDeviceTypeSizes = getKernel(
		programVerify, "verify_device_type_sizes");
	_kernelComputeMomentsFromLeafs = getKernel(
		programMoment, "compute_moments_from_leafs");
	_kernelComputeMomentsFromNodes = getKernel(
		programMoment, "compute_moments_from_nodes");
	_kernelFindInteractions = getKernel(
		programInteraction, "find_interactions");
	_kernelComputeInteractionIndices = getKernel(
		programInteraction, "compute_interaction_indices");
	_kernelComputeNodeMaxInteractionsLeafCount = getKernel(
		programInteraction, "compute_node_max_interactions_leaf_count");
	_kernelComputeLeafInteractionFields = getKernel(
		programField, "compute_leaf_interaction_fields");
	_kernelComputeNodeInteractionFields = getKernel(
		programField, "compute_node_interaction_fields");
	_kernelConvertLeafFieldsToForces = getKernel(
		programForce, "convert_leaf_fields_to_forces");
	_kernelConvertNodeFieldsToForces = getKernel(
		programForce, "convert_node_fields_to_forces");
	
	verifyDeviceTypeSizes();
}

void verifyDeviceTypeSize(
		std::string name,
		std::size_t deviceSize,
		std::size_t hostSize);

void OpenClSimulation::verifyDeviceTypeSizes() {
	// Read the sizes of the types on the device and verify that they are the
	// same as the sizes on the host.
	std::vector<cl_uint> sizes(VERIFY_NUM_TYPES);
	cl::Buffer sizesBuffer(
		_context,
		CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		sizeof(cl_uint) * sizes.size(),
		NULL);
	KernelData kernelData = _kernelVerifyDeviceTypeSizes;
	kernelData.kernel.setArg<cl::Buffer>(0, sizesBuffer);
	_queue.enqueueNDRangeKernel(
		kernelData.kernel,
		cl::NullRange,
		cl::NDRange(1));
	_queue.enqueueReadBuffer(
		sizesBuffer,
		CL_TRUE,
		0,
		sizeof(cl_uint) * sizes.size(),
		sizes.data());
	
	static_assert(
		sizeof(Octree::NodeInternal) == sizeof(device::node_t),
		"Octree::NodeInternal is not equivalent to node_t");
	static_assert(
		sizeof(Octree::LeafInternal) == sizeof(device::leaf_t),
		"Octree::LeafInternal is not equivalent to leaf_t");
	
	verifyDeviceTypeSize(
		"leaf_moment_t",
		sizes[VERIFY_LEAF_MOMENT_T_INDEX],
		sizeof(device::leaf_moment_t));
	verifyDeviceTypeSize(
		"node_moment_t",
		sizes[VERIFY_NODE_MOMENT_T_INDEX],
		sizeof(device::node_moment_t));
	verifyDeviceTypeSize(
		"leaf_value_t",
		sizes[VERIFY_LEAF_VALUE_T_INDEX],
		sizeof(device::leaf_value_t));
	verifyDeviceTypeSize(
		"node_value_t",
		sizes[VERIFY_NODE_VALUE_T_INDEX],
		sizeof(device::node_value_t));
	verifyDeviceTypeSize(
		"leaf_t",
		sizes[VERIFY_LEAF_T_INDEX],
		sizeof(device::leaf_t));
	verifyDeviceTypeSize(
		"node_t",
		sizes[VERIFY_NODE_T_INDEX],
		sizeof(device::node_t));
	verifyDeviceTypeSize(
		"leaf_field_t",
		sizes[VERIFY_LEAF_FIELD_T_INDEX],
		sizeof(device::leaf_field_t));
	verifyDeviceTypeSize(
		"node_field_t",
		sizes[VERIFY_NODE_FIELD_T_INDEX],
		sizeof(device::node_field_t));
	verifyDeviceTypeSize(
		"interaction_t",
		sizes[VERIFY_INTERACTION_T_INDEX],
		sizeof(device::interaction_t));
	
	_log << "Successfully verified all device types.\n";
}

void verifyDeviceTypeSize(
		std::string name,
		std::size_t deviceSize,
		std::size_t hostSize) {
	if (deviceSize != hostSize) {
		std::stringstream errorString;
		errorString <<
			"Type " << name <<
			" has size " << deviceSize <<
			" on device but different size " << hostSize << " on host";
		throw std::runtime_error(errorString.str());
	}
}

void OpenClSimulation::kernelComputeMomentsFromLeafs(
		device::BufferWrapper<device::leaf_t> leafs,
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::index_t> processedNodes) {
	// Pass the arguments to the kernel.
	KernelData kernelData = _kernelComputeMomentsFromLeafs;
	kernelData.kernel.setArg<device::index_t>(0, leafs.size());
	kernelData.kernel.setArg<cl::Buffer>(1, leafs.buffer());
	kernelData.kernel.setArg<device::index_t>(2, nodes.size());
	kernelData.kernel.setArg<cl::Buffer>(3, nodes.buffer());
	kernelData.kernel.setArg<cl::Buffer>(4, processedNodes.buffer());
	
	// Invoke the kernel.
	std::size_t numItems = nodes.size();
	std::size_t localSize = kernelData.workGroupSizeMultiple;
	std::size_t numWorkGroups =
		numItems / localSize +
		(numItems % localSize != 0) +
		(numItems == 0);
	std::size_t globalSize = numWorkGroups * localSize;
	_queue.enqueueNDRangeKernel(
		kernelData.kernel,
		cl::NullRange,
		cl::NDRange(globalSize),
		cl::NullRange);
}

void OpenClSimulation::kernelComputeMomentsFromNodes(
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::index_t> processedNodes,
		device::BufferWrapper<device::index_t> newProcessedNodes) {
	// Pass the arguments to the kernel.
	std::size_t numNodesToScan = 8;
	KernelData kernelData = _kernelComputeMomentsFromNodes;
	kernelData.kernel.setArg<device::index_t>(0, nodes.size());
	kernelData.kernel.setArg<cl::Buffer>(1, nodes.buffer());
	kernelData.kernel.setArg<device::index_t>(2, processedNodes.size());
	kernelData.kernel.setArg<cl::Buffer>(3, processedNodes.buffer());
	kernelData.kernel.setArg<cl::Buffer>(4, newProcessedNodes.buffer());
	kernelData.kernel.setArg<device::index_t>(5, numNodesToScan);
	
	// Invoke the kernel.
	std::size_t numItems =
		processedNodes.size() / numNodesToScan +
		(processedNodes.size() % numNodesToScan != 0);
	std::size_t localSize = kernelData.workGroupSizeMultiple;
	std::size_t numWorkGroups =
		numItems / localSize +
		(numItems % localSize != 0) +
		(numItems == 0);
	std::size_t globalSize = numWorkGroups * localSize;
	_queue.enqueueNDRangeKernel(
		kernelData.kernel,
		cl::NullRange,
		cl::NDRange(globalSize),
		cl::NullRange);
}

void OpenClSimulation::kernelFindInteractions(
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::interaction_t> interactions,
		device::BufferWrapper<device::interaction_t> newInteractions) {
	// Pass the arguments to the kernel.
	KernelData kernelData = _kernelFindInteractions;
	kernelData.kernel.setArg<device::index_t>(0, nodes.size());
	kernelData.kernel.setArg<cl::Buffer>(1, nodes.buffer());
	kernelData.kernel.setArg<device::index_t>(2, interactions.size());
	kernelData.kernel.setArg<cl::Buffer>(3, interactions.buffer());
	kernelData.kernel.setArg<cl::Buffer>(4, newInteractions.buffer());
	
	// Invoke the kernel.
	std::size_t numItems = interactions.size();
	std::size_t localSize = 8;
	std::size_t numWorkGroups = numItems + (numItems == 0);
	std::size_t globalSize = numWorkGroups * localSize;
	_queue.enqueueNDRangeKernel(
		kernelData.kernel,
		cl::NullRange,
		cl::NDRange(globalSize, localSize),
		cl::NDRange(localSize, localSize));
}

void OpenClSimulation::kernelComputeInteractionIndices(
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::interaction_t> interactions,
		device::BufferWrapper<device::index_t> nodeNumInteractions) {
	// Pass the arguments to the kernel.
	KernelData kernelData = _kernelComputeInteractionIndices;
	kernelData.kernel.setArg<device::index_t>(0, nodes.size());
	kernelData.kernel.setArg<cl::Buffer>(1, nodeNumInteractions.buffer());
	kernelData.kernel.setArg<device::index_t>(2, interactions.size());
	kernelData.kernel.setArg<cl::Buffer>(3, interactions.buffer());
	
	// Invoke the kernel.
	std::size_t numItems = interactions.size();
	std::size_t localSize = kernelData.workGroupSizeMultiple;
	std::size_t numWorkGroups = 
		numItems / localSize +
		(numItems % localSize != 0) +
		(numItems == 0);
	std::size_t globalSize = numWorkGroups * localSize;
	_queue.enqueueNDRangeKernel(
		kernelData.kernel,
		cl::NullRange,
		cl::NDRange(globalSize),
		cl::NullRange);
}

void OpenClSimulation::kernelComputeNodeMaxInteractionsLeafCount(
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::interaction_t> interactions,
		device::BufferWrapper<device::index_t> nodeMaxInteractionsLeafCount) {
	// Pass the arguments to the kernel.
	KernelData kernelData = _kernelComputeNodeMaxInteractionsLeafCount;
	kernelData.kernel.setArg<device::index_t>(0, nodes.size());
	kernelData.kernel.setArg<cl::Buffer>(1, nodes.buffer());
	kernelData.kernel.setArg<cl::Buffer>(2, nodeMaxInteractionsLeafCount.buffer());
	kernelData.kernel.setArg<device::index_t>(3, interactions.size());
	kernelData.kernel.setArg<cl::Buffer>(4, interactions.buffer());
	
	// Invoke the kernel.
	std::size_t numItems = interactions.size();
	std::size_t localSize = kernelData.workGroupSizeMultiple;
	std::size_t numWorkGroups = 
		numItems / localSize +
		(numItems % localSize != 0) +
		(numItems == 0);
	std::size_t globalSize = numWorkGroups * localSize;
	_queue.enqueueNDRangeKernel(
		kernelData.kernel,
		cl::NullRange,
		cl::NDRange(globalSize),
		cl::NullRange);
}

void OpenClSimulation::kernelComputeLeafInteractionFields(
		device::BufferWrapper<device::leaf_t> leafs,
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::interaction_t> leafInteractions,
		device::BufferWrapper<device::index_t> leafFieldIndices,
		device::BufferWrapper<device::index_t> nodeMaxInteractionsLeafCount,
		device::BufferWrapper<device::leaf_field_t> leafFields) {
	// Pass the arguments to the kernel.
	KernelData kernelData = _kernelComputeLeafInteractionFields;
	kernelData.kernel.setArg<device::index_t>(0, leafs.size());
	kernelData.kernel.setArg<cl::Buffer>(1, leafs.buffer());
	kernelData.kernel.setArg<cl::Buffer>(2, leafFieldIndices.buffer());
	kernelData.kernel.setArg<device::index_t>(3, nodes.size());
	kernelData.kernel.setArg<cl::Buffer>(4, nodes.buffer());
	kernelData.kernel.setArg<cl::Buffer>(5, nodeMaxInteractionsLeafCount.buffer());
	kernelData.kernel.setArg<device::index_t>(6, leafInteractions.size());
	kernelData.kernel.setArg<cl::Buffer>(7, leafInteractions.buffer());
	kernelData.kernel.setArg<device::index_t>(8, leafFields.size());
	kernelData.kernel.setArg<cl::Buffer>(9, leafFields.buffer());
	
	// Invoke the kernel.
	std::size_t numItems = leafInteractions.size();
	std::size_t localSize = 8;
	std::size_t numWorkGroups = numItems + (numItems == 0);
	std::size_t globalSize = numWorkGroups * localSize;
	_queue.enqueueNDRangeKernel(
		kernelData.kernel,
		cl::NullRange,
		cl::NDRange(globalSize, localSize),
		cl::NDRange(localSize, localSize));
}

void OpenClSimulation::kernelComputeNodeInteractionFields(
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::interaction_t> nodeInteractions,
		device::BufferWrapper<device::index_t> nodeFieldIndices,
		device::BufferWrapper<device::index_t> nodeNumNodeParentInteractions,
		device::BufferWrapper<device::node_field_t> nodeFields) {
	// Pass the arguments to the kernel.
	KernelData kernelData = _kernelComputeNodeInteractionFields;
	kernelData.kernel.setArg<device::index_t>(0, nodeFieldIndices.size());
	kernelData.kernel.setArg<cl::Buffer>(1, nodeFieldIndices.buffer());
	kernelData.kernel.setArg<device::index_t>(2, nodes.size());
	kernelData.kernel.setArg<cl::Buffer>(3, nodes.buffer());
	kernelData.kernel.setArg<cl::Buffer>(4, nodeNumNodeParentInteractions.buffer());
	kernelData.kernel.setArg<device::index_t>(5, nodeInteractions.size());
	kernelData.kernel.setArg<cl::Buffer>(6, nodeInteractions.buffer());
	kernelData.kernel.setArg<device::index_t>(7, nodeFields.size());
	kernelData.kernel.setArg<cl::Buffer>(8, nodeFields.buffer());
	
	// Invoke the kernel.
	std::size_t numItems = nodeInteractions.size();
	std::size_t localSize = kernelData.workGroupSizeMultiple;
	std::size_t numWorkGroups = numItems + (numItems == 0);
	std::size_t globalSize = 2 * numWorkGroups * localSize;
	_queue.enqueueNDRangeKernel(
		kernelData.kernel,
		cl::NullRange,
		cl::NDRange(globalSize),
		cl::NDRange(localSize));
}

void OpenClSimulation::kernelConvertLeafFieldsToForces(
		device::BufferWrapper<device::leaf_t> leafs,
		device::BufferWrapper<device::index_t> leafFieldIndices,
		device::BufferWrapper<device::leaf_field_t> leafFields,
		device::BufferWrapper<device::force_t> leafForces) {
	// Pass the arguments to the kernel.
	KernelData kernelData = _kernelConvertLeafFieldsToForces;
	kernelData.kernel.setArg<device::index_t>(0, leafs.size());
	kernelData.kernel.setArg<cl::Buffer>(1, leafs.buffer());
	kernelData.kernel.setArg<cl::Buffer>(2, leafFieldIndices.buffer());
	kernelData.kernel.setArg<cl::Buffer>(3, leafForces.buffer());
	kernelData.kernel.setArg<device::index_t>(4, leafFields.size());
	kernelData.kernel.setArg<cl::Buffer>(5, leafFields.buffer());
	
	// Invoke the kernel.
	std::size_t numItems = leafs.size();
	std::size_t localSize = kernelData.workGroupSizeMultiple;
	std::size_t numWorkGroups = 
		numItems / localSize +
		(numItems % localSize != 0) +
		(numItems == 0);
	std::size_t globalSize = numWorkGroups * localSize;
	_queue.enqueueNDRangeKernel(
		kernelData.kernel,
		cl::NullRange,
		cl::NDRange(globalSize),
		cl::NullRange);
}

void OpenClSimulation::kernelConvertNodeFieldsToForces(
		device::BufferWrapper<device::leaf_t> leafs,
		device::BufferWrapper<device::index_t> nodeFieldIndices,
		device::BufferWrapper<device::node_field_t> nodeFields,
		device::BufferWrapper<device::force_t> nodeForces) {
	// Pass the arguments to the kernel.
	KernelData kernelData = _kernelConvertNodeFieldsToForces;
	kernelData.kernel.setArg<device::index_t>(0, leafs.size());
	kernelData.kernel.setArg<cl::Buffer>(1, leafs.buffer());
	kernelData.kernel.setArg<cl::Buffer>(2, nodeFieldIndices.buffer());
	kernelData.kernel.setArg<cl::Buffer>(3, nodeForces.buffer());
	kernelData.kernel.setArg<device::index_t>(4, nodeFields.size());
	kernelData.kernel.setArg<cl::Buffer>(5, nodeFields.buffer());
	
	// Invoke the kernel.
	std::size_t numItems = leafs.size();
	std::size_t localSize = kernelData.workGroupSizeMultiple;
	std::size_t numWorkGroups = 
		numItems / localSize +
		(numItems % localSize != 0) +
		(numItems == 0);
	std::size_t globalSize = numWorkGroups * localSize;
	_queue.enqueueNDRangeKernel(
		kernelData.kernel,
		cl::NullRange,
		cl::NDRange(globalSize),
		cl::NullRange);
}

cl::Program OpenClSimulation::buildSourceFile(std::string fileName) {
	// Load the OpenCL source from file into a string.
	std::ifstream file(fileName);
	std::stringstream stream;
	stream << file.rdbuf();
	std::string source = stream.str();
	
	// Compile the source code.
	_log << "Build OpenCL source file " << fileName << ".\n";
	cl::Program program(_context, source);
	program.build();
	
	// Show the log in case there are warnings.
	std::string buildLog;
	program.getBuildInfo(_device, CL_PROGRAM_BUILD_LOG, &buildLog);
	if (!std::all_of(
			buildLog.begin(),
			buildLog.end(),
			[](char c) { return std::isspace(c); })) {
		_log << "Build log:\n";
		_log << buildLog << "\n";
	}
	
	return program;
}

OpenClSimulation::KernelData OpenClSimulation::getKernel(
		cl::Program const& program,
		std::string kernelName) {
	KernelData result;
	result.kernel = cl::Kernel(program, kernelName.c_str());
	result.kernel.getWorkGroupInfo(
		_device,
		CL_KERNEL_WORK_GROUP_SIZE,
		&result.maxWorkGroupSize);
	result.kernel.getWorkGroupInfo(
		_device,
		CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
		&result.compileWorkGroupSize);
	result.kernel.getWorkGroupInfo(
		_device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		&result.workGroupSizeMultiple);
	return result;
}

