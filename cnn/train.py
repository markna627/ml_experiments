
import data
import cnn_model
import torch
import torch.nn as nn
import argparse

def train(nepochs = 15):
  batched_training_images, batched_training_labels, batched_validation_images, batched_validation_labels = data.datasets()
  model = cnn_model.CNN()
  metrics = {'training_loss': [],
             'training_accuracy': [],
             'validation_loss': [],
             'validation_accuracy': []}
  nepochs = nepochs
  lr = 10e-3

  optimizer = torch.optim.Adam(model.parameters(), lr)
  loss_fn = nn.CrossEntropyLoss()


  def accuracy(pred, label):
    model_pred = torch.squeeze(pred.argmax(dim = 1))
    correct = len(pred) - torch.count_nonzero(label - model_pred)
    accuracy = correct/len(pred)
    return accuracy

  for epoch in range(nepochs):
    model.train()
    train_loss = 0 
    for step, (mini_batch, training_label) in enumerate(zip(batched_training_images,batched_training_labels)):
      pred = model(mini_batch)
      loss = loss_fn(pred, training_label)
        
      train_loss += loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      metrics['training_loss'].append(loss.item())
      metrics['training_accuracy'].append(accuracy(pred, training_label))
    print(f'Epoch: {epoch} - Training Loss: {train_loss/len(batched_training_images):.3f}')

    model.eval()
    with torch.no_grad():
      for step, (valid_image, valid_label) in enumerate(zip(batched_validation_images, batched_validation_labels)):
        pred = model(valid_image)
        validation_loss = loss_fn(pred, valid_label)
        metrics['validation_loss'].append(validation_loss.item())
        metrics['validation_accuracy'].append(accuracy(pred, valid_label))
  return metrics

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--nepochs", type = int, default = 15)
  args = parser.parse_args()
  train(args.nepochs)
