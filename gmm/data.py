
import pandas as pd
import torch
'''
Synthetic data
'''
def load_data():
	mu1 = torch.tensor([2.0, 1.0])
	cov1 = torch.tensor([
	    [1, 0.7],
	    [0.7, 1]
	])
	mu2 = torch.tensor([4.0, 6.0])
	cov2 = torch.tensor([
	    [1.0, 0.4],
	    [0.4, 1.0]
	])
	mu3 = torch.tensor([7.0, 3.0 ])
	cov3 = torch.tensor([
	    [2.0, 0.2],
	    [0.2, 2.0]
	])
	num_sample = 200
	dist1 = torch.distributions.MultivariateNormal(mu1, covariance_matrix = cov1)
	dist2 = torch.distributions.MultivariateNormal(mu2, covariance_matrix = cov2)
	dist3 = torch.distributions.MultivariateNormal(mu3, covariance_matrix = cov3)

	sample1 = dist1.sample((num_sample,))
	sample2 = dist2.sample((num_sample,))
	sample3 = dist3.sample((num_sample,))

	return torch.vstack([sample1, sample2, sample3])