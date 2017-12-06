#ifndef __NBODY_SIMULATION_H_
#define __NBODY_SIMULATION_H_

namespace nbody {

template<typename TScalar, typename TVector>
class Simulation {
	
private:
	
public:
	
	using Scalar = TScalar;
	using Vector = TVector;
	
	struct Particle final {
		Vector position;
		Vector velocity;
		Scalar mass;
		Scalar charge;
		
		Particle(
				Vector position,
				Vector velocity,
				Scalar mass,
				Scalar charge) :
				position(position),
				velocity(velocity),
				mass(mass),
				charge(charge) {
		}
	};
	
	virtual Scalar step() = 0;
	virtual std::vector<Particle> particles() const = 0;
	
};

}

#endif

