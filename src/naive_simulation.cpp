#include "nbody/naive_simulation.h"

#include <cmath>

using namespace nbody;

NaiveSimulation::Scalar NaiveSimulation::step() {
	for (unsigned int i = 1; i < _particles.size(); ++i) {
		for (unsigned int j = 0; j < i; ++j) {
			Vector delta = {
				_particles[j].position[0] - _particles[i].position[0],
				_particles[j].position[0] - _particles[i].position[0],
				_particles[j].position[0] - _particles[i].position[0]
			};
			
			Scalar r = std::sqrt(
				delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
			Scalar chargeFactor = _particles[i].charge * _particles[j].charge;
			Scalar massA = _particles[i].mass;
			Scalar massB = _particles[j].mass;
			
			Vector force = {
				_forceConstant * chargeFactor * delta[0] / (r * r * r),
				_forceConstant * chargeFactor * delta[1] / (r * r * r),
				_forceConstant * chargeFactor * delta[2] / (r * r * r)
			};
			
			_particles[i].velocity[0] += force[0] / massA * _timeStep;
			_particles[i].velocity[1] += force[1] / massA * _timeStep;
			_particles[i].velocity[2] += force[2] / massA * _timeStep;
			
			_particles[j].velocity[0] -= force[0] / massB * _timeStep;
			_particles[j].velocity[1] -= force[1] / massB * _timeStep;
			_particles[j].velocity[2] -= force[2] / massB * _timeStep;
		}
	}
	
	for (unsigned int i = 0; i < _particles.size(); ++i) {
		_particles[i].position[0] += _particles[i].velocity[0] * _timeStep;
		_particles[i].position[1] += _particles[i].velocity[1] * _timeStep;
		_particles[i].position[2] += _particles[i].velocity[2] * _timeStep;
	}
	
	_time += _timeStep;
	return _time;
}

