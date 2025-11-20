import numpy as np
import random

def global_energy(state, weight, bias):
  '''
  input:
  - state: an array of N neurons with current states
  - weight: N x N 2D array of weights between nueron i and neuron j
  - bias: an array of thresholds corresponding to each neuron we update the nueron based on

  output:
  - energy strain calculated based on the paper
  '''
  return -np.dot(np.dot(state,weight),state)/2 - np.dot(bias,state)

def update(state, weight, bias):
  '''
  A randomly chosen neuron gets updated given weights and bias

  input
  - state: an array of N neurons with current states
  - weight: N x N 2D array of weights between nueron i and neuron j
  - bias: an array of thresholds corresponding to each neuron we update the nueron based on

  output:
  - updated_state: an array of N neurons with the now-updated neuron
  '''
  index = random.randint(0, len(state)-1)
  updated_state = state.copy()

  T = np.dot(state, weight[index])

  if T > bias[index]:
    updated_state[index] = 1
  else:
    updated_state[index] = -1
  return updated_state

def run_updates(state, weight, bias, n_iter = 200, steps = None):
  '''
  input:
  - state: an array of N neurons with current states
  - weight: N x N 2D array of weights between nueron i and neuron j
  - bias: an array of thresholds corresponding to each neuron we update the nueron based on
  - n_iter: an integer number of iterations to run the updates
  - steps: steps we want to see the updated state's energy for
  output:
  - state (list[]): final state after the updates
  - energy_trends (list): energy value for every step
  - states (list[]): snapshots of states evolution
  '''
  state = state.copy()
  energy_trends = np.zeros(n_iter)
  states = []
  for i in range(n_iter):
    state = update(state, weight, bias)
    if (steps is not None) and (i%steps) == 0:
      energy_trends[i] = global_energy(state,weight,bias)
      states.append(state.copy())
  return state, energy_trends, states



