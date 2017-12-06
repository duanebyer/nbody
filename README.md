# nbody
Simple n-body gravitatational simulation using the fast multipole method (FMM).

## Introduction
This project aimed to implement the fast multipole method on the GPU using
octrees to spatially partition the particles. In the end, it wasn't very
successful because octrees are difficult to iterate over using thousands of
threads. Invariably, some of the GPU threads end up unused. Furthermore, in a
(perhaps misguided) attempt to avoid GPU atomic operations at all costs, the
method that this project used for computing the interactions between the
particles was:

1. Compute the interactions between nodes/particles that need to be evaluated.
   Each particle is generally part of many, many interactions.
2. Actually evaluate the forces from those interactions into a fixed buffer of
   memory (which can only be allocated after knowing how many interactions each
   particle is part of).
3. Finally reducing those forces for each particle in turn.
4. Integrating the forces to determine the motion of the particles.

This approach is not ideal for many reasons. It requires large amounts of data
to be constantly streamed between the host and the device, as well as large
amounts of data to stored on the GPU. Also, this implementation takes what is in
principle a very simple recursive algorithm and splits it so that it first
calculates what calculations need to be done. The layer of indirection is
probably a huge performance killer.

I would expect that a simple CPU multithreaded solution could easily outperform
this solution. It's probably possible to efficiently use octrees on the GPU, but
I certainly haven't figured out how to do it here.

## Fast multipole method
The fast multipole method works on a set of data points contained in an octree.
Starting with the root node, the algorithm can be applied recursively until the
gravitational forces within the whole octree have been calculated.

### Simple recursive definition
The algorithm can be defined in a recursive way. Consider a set of nodes (for
instance, from an octree). To calculate the gravitational forces on the
particles contained within the nodes, loop through each possible pair of nodes.

For a given pair, if the two nodes are far enough apart, then evaluate the
multipole moment of each of the nodes. Approximate the gravitational field at
each node due to the other node by using a Taylor approximation (from the
multipole moment). Apply this field to each of the data points within the two
nodes.

If the two nodes are too close for this approximation to be reasonable, then
take all of the child nodes of both nodes and recursively call this algorithm on
the complete set of child nodes.

