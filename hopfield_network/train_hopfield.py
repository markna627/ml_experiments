import numpy as np
import random
import hopfield
import argparse

three = np.array([
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
	[0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
	[0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
	[0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
	[0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0],
	[0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
	[0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	], dtype=int)
	#a 3 produced by ChatGPT


def train(corruption_rate, n_iter):
	v = three.flatten()
	v = 2*v - 1 #binarizing the input
	w = np.outer(v, v) #memory mounted through the weights
	np.fill_diagonal(w, 0)
	bias = np.zeros(256)

	num_flips = int(len(v) * corruption_rate)  # e.g. flip 20% of the pixels to produce corrupted input
	flip_indices = np.random.choice(len(v), size=num_flips, replace=False)
	v[flip_indices] *= -1

	state, energy_trends, states = hopfield.run_updates(v,w,bias, n_iter = n_iter, steps = 100)
	return state, energy_trends, states

def visualize(state):
	for row in state.reshape(16,16):
		print("".join("██" if x == 1 else "  " for x in row))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--corruption_rate", type = float, default = 0.2, help = "Corruption rate to damage the image.")
	parser.add_argument("--n_iter", type = int, default = 1000, help = "Number of iteration to recover the image")
	args = parser.parse_args()
	state, energy_trends, states = train(args.corruption_rate, args.n_iter)


	print("Initial state give: \n")
	visualize(states[0])
	# print(f'Initial state given: \n {states[0].reshape(16,16)}')
	print()
	print('Final state recovered: \n')
	visualize(state)

