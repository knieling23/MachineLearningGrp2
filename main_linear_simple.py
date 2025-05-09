import numpy as np
from criterion import softmax_cross_entropy_loss
from dataloaders import NonLinearDataset, LinearDataset
from optimizer import bgd
from network import linearClassifier

"""
Module that runs the training of the linear classifier.
"""


def train_step(dataloader, network, criterion, optimizer):
    """
    Function that calculates one training step.

    Parameter:
      dataloader - a class that holds the training data
      network - class of the network to train
      criterion - class implementing the loss function
      optimizer - class implementing the optimizer
    """
    X = dataloader.data_input
    y = dataloader.data_output
    # Forward
    predictions = network.forward(X)
    # Loss berechnen
    loss = criterion.calc_loss(predictions, y, network.weights)
    # Gradient Data Loss
    grad_data = criterion.gradient_data_loss(predictions, y)
    # Gradient Reg Loss
    grad_reg = criterion.gradient_reg(network.weights)
    # Backprop
    grad_w = network.backprop_layer_weights(X, grad_data) + grad_reg
    grad_b = network.backprop_layer_bias(grad_data)
    # Update Parameter
    network.weights = optimizer.step(grad_w, network.weights)
    network.bias = optimizer.step(grad_b, network.bias)
    return loss


def train(epochs, dataloader, network, criterion, optimizer):

    for i in range(epochs):
        train_loss = train_step(dataloader, network, criterion, optimizer)
        if i % (epochs // 10) == 0:
            print("iteration %d: loss %f" % (i, train_loss))

    dataloader.eval_network(network)
    dataloader.display_classification_result(network)
            
def main():
    step_size = 1e-0
    reg = 1e-3
    epoch = 200

    data = NonLinearDataset()
    data = LinearDataset()
    
    optim = bgd(step_size)
    
    net = linearClassifier(data.Dim, data.K)
    
    crit = softmax_cross_entropy_loss(reg)

    train(epoch, data, net, crit, optim)


if __name__ == "__main__":
    np.random.seed(42)
    main()
