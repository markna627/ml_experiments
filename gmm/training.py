import gmm
import data
import torch
def train():
  num_epochs = 50
  components = 3
  x = data.load_data()
  print(f'Samples Used: \n Mean: \n{[2.0, 1.0]}, \n{[4.0, 6.0]}, \n{[7.0, 3.0 ]}')
  print(f'Variances:\n {[[1.0, 0.7],[0.7, 1]]},\n {[[1.0, 0.4],[0.4, 1.0]]},\n {[[2.0, 0.2],[0.2, 2.0]]}')

  thres = 1e-5
  prev_likelihood = 0
  prev_l = 0.0
  curr_l = 0.0
  pi = torch.tensor([0.6,0.2,0.2]) #random probability
  mu = torch.tensor([[1,1], [1,2], [3, 2]]) # random points from the dataset
  Sigma = torch.stack([torch.eye(2), 2 * torch.eye(2), 1.5 * torch.eye(2)])
  model = gmm.GMM(components, x)

  for epoch in range(num_epochs):
    responsibility = model.forward(pi, mu, Sigma)
    pi, mu, Sigma = model.M_step(responsibility)
    curr_likelihood = model.get_log_likelihood(pi, mu, Sigma)
    curr_l = curr_likelihood - prev_likelihood
    print(f'Epoch {epoch} - Log-likelihood: {curr_l:.4f}')
    if abs(curr_l - prev_l) < thres:
      break
    prev_l = curr_l
  pi, mu, Sigma = model.stats()

  print(f'Estimated weights for each cluster:\n {[float(weight) for weight in pi]}\n')
  print(f'\nEstimated means of each cluster:\n {mu}')
  print(f'\nEstimated covariances for each cluster: \n{Sigma}\n')
  return model.stats()


if __name__ == "__main__":
  print('run')
  train()
