#ifndef __NBODY_NAIVE_SIMULATION_H_
#define __NBODY_NAIVE_SIMULATION_H_

#include <vector>

#include "nbody/device/types.h"

#include "nbody/simulation.h"

namespace nbody {

class NaiveSimulation final :
		public Simulation<device::scalar_t, device::vector_t> {
	
private:
	
	std::vector<Particle> _particles;
	
	Scalar _forceConstant;
	Scalar _time;
	Scalar _timeStep;
	
public:
	
	NaiveSimulation(
			std::vector<Particle> particles,
			Scalar forceConstant,
			Scalar timeStep) :
			_particles(particles),
			_forceConstant(forceConstant),
			_time(0),
			_timeStep(timeStep) {
	}
	
	Scalar step() override;
	
	std::vector<Particle> particles() const override {
		return std::vector<Particle>(_particles);
	}
	
};

}

#endif

