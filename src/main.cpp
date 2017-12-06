#include <cstdlib>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "nbody/animation.h"
#include "nbody/naive_simulation.h"
#include "nbody/open_cl_simulation.h"

using Simulation = nbody::OpenClSimulation;

Simulation::Scalar uniformRandom();

int main(int argc, char** argv) {
	try {
		unsigned int seed = std::time(NULL);
		std::srand(seed);
		std::cout << "Using random number generator seed " << seed << ".\n";
		
		// Parameters for simulation initial state.
		unsigned int numParticles = 1000000;
		Simulation::Vector bounds = { 1.0, 1.0, 1.0, 0.0 };
		Simulation::Scalar velocityMax = 0.1;
		Simulation::Scalar massRange[2] = { 1.0, 10.0 };
		Simulation::Scalar chargeRange[2] = { 0.1, 1.0 };
		
		// Create randomly positioned particles.
		std::cout << "Generating particles.\n";
		std::vector<Simulation::Particle> particles;
		particles.reserve(numParticles);
		for (unsigned int i = 0; i < numParticles; ++i) {
			Simulation::Vector position = {
				bounds[0] * (uniformRandom()),
				bounds[1] * (uniformRandom()),
				bounds[2] * (uniformRandom()),
				0.0
			};
			Simulation::Scalar theta =
				2.0 * M_PI * uniformRandom();
			Simulation::Scalar phi =
				std::acos(2.0 * (uniformRandom() - 0.5));
			Simulation::Vector velocity = {
				velocityMax * std::sin(phi) * std::cos(theta),
				velocityMax * std::sin(phi) * std::sin(theta),
				velocityMax * std::cos(phi),
				0.0
			};
			Simulation::Scalar massFraction = uniformRandom();
			Simulation::Scalar mass =
				massRange[0] * (1.0 - massFraction) +
				massRange[1] * massFraction;
			
			Simulation::Scalar chargeFraction = uniformRandom();
			Simulation::Scalar charge =
				chargeRange[0] * (1.0 - chargeFraction) +
				chargeRange[1] * chargeFraction;
			Simulation::Particle particle = {
				position,
				velocity,
				mass,
				charge
			};
			particles.push_back(particle);
		}
		
		// Create the simulation.
		Simulation::Scalar timeStep = 0.001;
		Simulation simulation(
			bounds,
			particles,
			timeStep,
			std::cout);
		
		// Create a .CSV file to store the data in.
		std::ofstream dataFile("particles.csv");
		
		std::cout << "Starting simulation.\n";
		std::size_t stepIndex = 0;
		Simulation::Scalar time = 0.0;
		Simulation::Scalar maxTime = 0.01;
		while (time < maxTime) {
			// Take a step.
			time = simulation.step();
			
			// Output to data file.
			dataFile << time;
			for (Simulation::Particle p : simulation.particles()) {
				dataFile <<
					"," << p.position[0] <<
					"," << p.position[1] <<
					"," << p.position[2];
			}
			dataFile << "\n";
			
			// Update the counter.
			++stepIndex;
		}
		
		dataFile.close();
	}
	catch (cl::BuildError error) {
		std::cerr << "OpenCL build error " << error.err() <<
			" in " << error.what() << ":\n";
		for (auto pair : error.getBuildLog()) {
			std::cerr << std::get<std::string>(pair) << "\n";
		}
	}
	catch (cl::Error error) {
		std::cerr << "OpenCL error " << error.err() <<
			" in " << error.what() << "\n";
	}
	
	return 0;
}

Simulation::Scalar uniformRandom() {
	Simulation::Scalar rand =
		static_cast<Simulation::Scalar>(std::rand());
	return rand / RAND_MAX;
}

