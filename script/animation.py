import sys
import csv

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as anim

def main():
	if len(sys.argv) != 2:
		print "Expecting data file name, none provided."
		return
	particles = []
	with open(sys.argv[1], 'rb') as data_file:
		data_reader = csv.reader(data_file)
		for row in data_reader:
			if not row:
				continue
			particles.append(map(float, row[1:]))
	particles = np.array(particles)
	num_particles = np.shape(particles)[1] / 3
	particles = np.reshape(particles, (-1, num_particles, 3))
	
	fig = plt.figure()
	ax = p3.Axes3D(fig)
	ax.set_xlim3d([0.0, 1.0])
	ax.set_xlabel('X')
	ax.set_ylim3d([0.0, 1.0])
	ax.set_ylabel('Y')
	ax.set_zlim3d([0.0, 1.0])
	ax.set_zlabel('Z')
	
	points = ax.scatter(
		particles[0,:,0], particles[0,:,1], particles[0,:,2])
	
	def animate(i):
		xs = particles[i,:,0]
		ys = particles[i,:,1]
		zs = particles[i,:,2]
		points._offsets3d = (xs, ys, zs)
	
	num_frames = np.shape(particles)[0] - 1
	animation = anim.FuncAnimation(
		fig, animate, interval=10, frames=num_frames)
	
	plt.draw()
	plt.show()

if __name__ == "__main__":
	main()

