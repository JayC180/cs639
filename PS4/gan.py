from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of random noise from Gaussian distribution.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing
      noise from a Gaussian distribution.
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    noise = torch.randn(batch_size, noise_dim, device=device)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    model = []
    alpha = 0.05
    model.append(nn.Linear(784, 400))
    model.append(nn.LeakyReLU(negative_slope=alpha))
    model.append(nn.Linear(400, 200))
    model.append(nn.LeakyReLU(negative_slope=alpha))
    model.append(nn.Linear(200, 100))
    model.append(nn.LeakyReLU(negative_slope=alpha))
    model.append(nn.Linear(100, 1))
    model = nn.Sequential(*model)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################
    model = []
    model.append(nn.Linear(noise_dim, 128))
    model.append(nn.ReLU())
    model.append(nn.Linear(128, 256))
    model.append(nn.ReLU())
    model.append(nn.Linear(256, 512))
    model.append(nn.ReLU())
    model.append(nn.Linear(512, 784))
    model.append(nn.Tanh())
    model = nn.Sequential(*model)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    ones = torch.ones_like(logits_real)
    zeros = torch.zeros_like(logits_fake)
    bce_loss1 = nn.functional.binary_cross_entropy_with_logits(
        logits_real, ones)
    bce_loss2 = nn.functional.binary_cross_entropy_with_logits(
        logits_fake, zeros)
    loss = bce_loss1 + bce_loss2
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    ones = torch.ones_like(logits_fake)
    loss = nn.functional.binary_cross_entropy_with_logits(logits_fake, ones)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.5, 0.999))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def run_a_gan(D, G, D_solver, G_solver, loader_train, discriminator_loss, generator_loss, device, show_images, plt, show_every=250,
              batch_size=128, noise_size=96, num_epochs=10):
    """
    Train a GAN!

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - loader_train: the dataset used to train GAN
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            ##############################################################################
            # TODO: Implement an iteration of training the discriminator.                #
            # Replace 'pass' with your code.                                             #
            # Save the overall discriminator loss in the variable 'd_total_error',       #
            # which will be printed after every 'show_every' iterations.                 #
            #                                                                            #
            # IMPORTANT: make sure to pre-process your real data (real images),          #
            # so as to make it in the range [-1,1].                                      #
            ##############################################################################
            x -= x.min()
            x /= x.max()
            x = 2*x - 1
            B, C, H, W = x.shape
            x = x.reshape((B, -1)).to(device)
            d_total_error = None
            noise = sample_noise(batch_size=batch_size, noise_dim=noise_size, device=device)
            generated_image = G(noise)
            logits_real = D(x)
            logits_fake = D(generated_image)
            D_solver.zero_grad()
            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward(retain_graph=True)
            D_solver.step()
            ##############################################################################
            #                              END OF YOUR CODE                              #
            ##############################################################################

            ##############################################################################
            # TODO: In the same iteration, implement training of the generator now   .   #
            # Replace 'pass' with your code.                                             #
            # Save the generator loss in the variable 'g_error', which will be printed.  #
            # after every 'show_every' iterations, and save the fake images generated    #
            # by G in the variable 'fake_images', which will be used to visualize the    #
            # generated images.
            ##############################################################################
            g_error = None
            fake_images = generated_image
            G_solver.zero_grad()
            g_error = generator_loss(logits_fake)
            g_error.backward()
            G_solver.step()

            ##############################################################################
            #                              END OF YOUR CODE                              #
            ##############################################################################

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(
                    iter_count, d_total_error.item(), g_error.item()))
                imgs_numpy = fake_images.data.cpu()  # .numpy()
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1
        if epoch == num_epochs - 1:
            return imgs_numpy


def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    model = []
    model.append(nn.Unflatten(1, (1, 28,28)))
    model.append(nn.Conv2d(1, 32, 5))
    model.append(nn.LeakyReLU(negative_slope=0.01))
    model.append(torch.nn.MaxPool2d(2, stride=2))
    model.append(nn.Conv2d(32, 64, 5))
    model.append(nn.LeakyReLU(negative_slope=0.01))
    model.append(torch.nn.MaxPool2d(2, stride=2))
    model.append(nn.Flatten())
    model.append(nn.Linear(1024, 1024))
    model.append(nn.LeakyReLU(negative_slope=0.01))
    model.append(nn.Linear(1024, 1))
    model = nn.Sequential(*model)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    model = []
    model.append(nn.Linear(96,1024))
    model.append(nn.ReLU())
    model.append(nn.BatchNorm1d(1024))
    model.append(nn.Linear(1024, 6272))
    model.append(nn.ReLU())
    model.append(nn.BatchNorm1d(6272))
    model.append(nn.Unflatten(1, (128,7,7)))
    model.append(torch.nn.ConvTranspose2d(128, out_channels=64, kernel_size=4, stride=2, padding=1))
    model.append(nn.ReLU())
    model.append(nn.BatchNorm2d(64))
    model.append(torch.nn.ConvTranspose2d(64, out_channels=1, kernel_size=4, stride=2, padding=1))
    model.append(nn.Tanh())
    model.append(nn.Flatten(start_dim=1))
    model = nn.Sequential(*model)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
