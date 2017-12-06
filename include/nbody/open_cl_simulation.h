#ifndef __NBODY_OPEN_CL_SIMULATION_H_
#define __NBODY_OPEN_CL_SIMULATION_H_

#include <ostream>
#include <string>
#include <vector>

#include <glade/glade.h>

#include "nbody/device/cl_includes.h"

#include "nbody/device/buffer_wrapper.h"
#include "nbody/device/types.h"

#include "nbody/simulation.h"

namespace nbody {

class OpenClSimulation final :
		public Simulation<device::scalar_t, device::vector_t> {
	
private:
	
	// Contains a kernel together with information about work group sizes.
	struct KernelData {
		cl::Kernel kernel;
		std::size_t maxWorkGroupSize;
		std::size_t compileWorkGroupSize[3];
		std::size_t workGroupSizeMultiple;
	};
	
	// Octree and simulation data.
	struct OctreeInternalDetails {
		template<typename T>
		using VectorType = std::vector<T>;
		template<typename T>
		using SizeType = device::index_t;
		template<typename T>
		using DifferenceType = device::index_diff_t;
	};
	using Octree = glade::Orthtree<
		3,
		device::vector_t,
		device::leaf_value_t,
		device::node_value_t,
		OctreeInternalDetails>;
	
	Octree _octree;
	Scalar _time;
	Scalar _timeStep;
	
	std::ostream& _log;
	
	// Global OpenCL objects needed by everything.
	cl::Platform _platform;
	cl::Device _device;
	cl::Context _context;
	cl::CommandQueue _queue;
	
	cl_ulong _deviceMaxBufferSize;
	
	// OpenCL kernels.
	KernelData _kernelVerifyDeviceTypeSizes;
	KernelData _kernelComputeMomentsFromLeafs;
	KernelData _kernelComputeMomentsFromNodes;
	KernelData _kernelFindInteractions;
	KernelData _kernelComputeInteractionIndices;
	KernelData _kernelComputeNodeMaxInteractionsLeafCount;
	KernelData _kernelComputeLeafInteractionFields;
	KernelData _kernelComputeNodeInteractionFields;
	KernelData _kernelConvertLeafFieldsToForces;
	KernelData _kernelConvertNodeFieldsToForces;
	
	// Wrapper functions for the kernels to make it easier to use them.
	void verifyDeviceTypeSizes();
	void kernelComputeMomentsFromLeafs(
		device::BufferWrapper<device::leaf_t> leafs,
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::index_t> processedNodes);
	void kernelComputeMomentsFromNodes(
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::index_t> processedNodes,
		device::BufferWrapper<device::index_t> newProcessedNodes);
	void kernelFindInteractions(
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::interaction_t> interactions,
		device::BufferWrapper<device::interaction_t> newInteractions);
	void kernelComputeInteractionIndices(
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::interaction_t> interactions,
		device::BufferWrapper<device::index_t> nodeNumInteractions);
	void kernelComputeNodeMaxInteractionsLeafCount(
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::interaction_t> interactions,
		device::BufferWrapper<device::index_t> nodeMaxInteractionsLeafCount);
	void kernelComputeLeafInteractionFields(
		device::BufferWrapper<device::leaf_t> leafs,
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::interaction_t> leafInteractions,
		device::BufferWrapper<device::index_t> leafFieldIndices,
		device::BufferWrapper<device::index_t> nodeMaxInteractionsLeafCount,
		device::BufferWrapper<device::leaf_field_t> leafFields);
	void kernelComputeNodeInteractionFields(
		device::BufferWrapper<device::node_t> nodes,
		device::BufferWrapper<device::interaction_t> nodeInteractions,
		device::BufferWrapper<device::index_t> nodeFieldIndices,
		device::BufferWrapper<device::index_t> nodeNumNodeParentInteractions,
		device::BufferWrapper<device::node_field_t> nodeFields);
	void kernelConvertLeafFieldsToForces(
		device::BufferWrapper<device::leaf_t> leafs,
		device::BufferWrapper<device::index_t> leafFieldIndices,
		device::BufferWrapper<device::leaf_field_t> leafFields,
		device::BufferWrapper<device::force_t> leafForces);
	void kernelConvertNodeFieldsToForces(
		device::BufferWrapper<device::leaf_t> leafs,
		device::BufferWrapper<device::index_t> nodeFieldIndices,
		device::BufferWrapper<device::node_field_t> nodeFields,
		device::BufferWrapper<device::force_t> nodeForces);
	
	// Convenience methods for interfacing with OpenCL.
	void initialize();
	cl::Program buildSourceFile(std::string fileName);
	KernelData getKernel(cl::Program const& program, std::string kernelName);
	
	// Structures that hold buffers from intermediate computations.
	struct UnprocessedInteractionBuffers {
		std::vector<device::interaction_t> interactions = {
			{
				0, 0,
				0, 0,
				false, true
			}
		};
		std::vector<device::interaction_t> leafInteractions;
		std::vector<device::interaction_t> nodeInteractions;
		bool finished() const {
			return
				interactions.empty() &&
				leafInteractions.empty() &&
				nodeInteractions.empty();
		}
	};
	struct OctreeBuffers {
		device::BufferWrapper<device::leaf_t> leafs;
		device::BufferWrapper<device::node_t> nodes;
	};
	struct InteractionBuffers {
		device::BufferWrapper<device::interaction_t> leafInteractions;
		device::BufferWrapper<device::interaction_t> nodeInteractions;
		device::BufferWrapper<device::index_t> nodeNumLeafInteractions;
		device::BufferWrapper<device::index_t> nodeNumNodeInteractions;
		device::BufferWrapper<device::index_t> nodeMaxInteractionsLeafCount;
	};
	struct ForceBuffers {
		device::BufferWrapper<device::force_t> leafForces;
		device::BufferWrapper<device::force_t> nodeForces;
	};
	struct IntegrationBuffers {
		std::vector<device::vector_t> newVelocities;
		std::vector<device::vector_t> newPositions;
	};
	
	template<typename T>
	device::BufferWrapper<T> createBuffer(
			device::IOFlag flag,
			std::size_t size,
			T const* data = NULL) {
		return device::BufferWrapper<T>(_context, _queue, flag, size, data);
	}
	
	OctreeBuffers computeOctreeBuffers();
	InteractionBuffers computeInteractionBuffers(
		OctreeBuffers octreeBuffers,
		UnprocessedInteractionBuffers& unprocessed);
	ForceBuffers computeForceBuffers(
		OctreeBuffers octreeBuffers,
		InteractionBuffers interactionBuffers);
	IntegrationBuffers computeIntegrationBuffers(
		ForceBuffers forceBuffers,
		IntegrationBuffers initialIntegrationBuffers);
	void updateOctree(IntegrationBuffers integrationBuffers);
	
	device::index_t computeLeafFieldIndices(
		device::BufferWrapper<device::index_t> nodeNumNodeInteractions,
		device::BufferWrapper<device::index_t> nodeMaxInteractionsLeafCount,
		device::BufferWrapper<device::index_t> leafFieldIndices);
	device::index_t computeNodeFieldIndices(
		device::BufferWrapper<device::index_t> nodeNumLeafInteractions,
		device::BufferWrapper<device::index_t> nodeNumNodeParentInteractions,
		device::BufferWrapper<device::index_t> nodeFieldIndices);
	
public:
	
	OpenClSimulation(
		device::vector_t bounds,
		std::vector<Particle> particles,
		Scalar timeStep,
		std::ostream& log);
	
	Scalar step() override;
	std::vector<Particle> particles() const override;
	
};

}

#endif

