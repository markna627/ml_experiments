
import torch
import torch.nn as nn

class GMM(nn.Module):
  def __init__(self, k, x):
    super().__init__()
    self.k = k
    self.x = x
    self.x_num = x.shape[0]

    self.pi = 0
    self.mu = 0
    self.Sigma = 0
  def M_step(self, responsibility):
    '''
    input:
    - responsibility: N x K tensor of soft assignment of data
    output:
    - pi_k: (1,K) tensor of estimated weights of the distributions
    - mu_k: (K, 1, D) tensor of average of the data for each cluster
    - Sigma_k: (K, D, D) tensor of covariance of each cluster
    '''
    effective_assign_k = torch.sum(responsibility, dim = 0)
    pi_k = effective_assign_k / self.x_num
    mu_k = torch.sum(responsibility.T.unsqueeze(-1) * self.x, dim = 1)/effective_assign_k.unsqueeze(-1)
    mean_diff = self.x - mu_k.unsqueeze(1)
    Sigma_k = torch.einsum('kn, knd, kne -> kde', responsibility.T, mean_diff, mean_diff)
    Sigma_k = Sigma_k/effective_assign_k.unsqueeze(-1).unsqueeze(-1)
    #effective_assign_k = (K x D)
    self.mu, self.Sigma, self.pi = mu_k, Sigma_k, pi_k
    return pi_k, mu_k, Sigma_k
  def forward(self, pi, mu, Sigma): #E_step
    '''
    input:
    - pi: (1,K) tensor of estimated weights of the distributions
    - mu: (K, 1, D) tensor of average of the data for each cluster
    - Sigma: (K, D, D) tensor of covariance of each cluster
    output:
    - res: N x K tensor of soft assignment of data
    '''
    dists = [torch.distributions.MultivariateNormal(mu[k], covariance_matrix = Sigma[k]) for k in range(self.k)]
    res_k = torch.stack([dist.log_prob(self.x) for dist in dists]).T.exp() * pi
    normalization = torch.sum(res_k, dim = 1).unsqueeze(1)
    res = res_k/normalization # N x K
    return res

  def get_log_likelihood(self, pi, mu, Sigma):
    dists = [torch.distributions.MultivariateNormal(mu[k], Sigma[k]) for k in range( self.k)]
    log_probs = torch.stack([dist.log_prob(self.x) for dist in dists], dim = 1)
    weighted = log_probs.exp() * pi
    L = torch.sum(torch.log(torch.sum(weighted, dim=0)))
    return L
  def stats(self):
  	return self.pi, self.mu, self.Sigma

