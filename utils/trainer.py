"""
Class that holds the main train loop, for a specific part of the training progress
"""
import torch.nn as nn
import settings

import torch

from utils.utils import cyclic_data_iterator

from torch.autograd import Variable


class StageTrainer:

    def __init__(self, G, D, opt_G, opt_D, data_loader, stage=6, conversion_depth=16):
        self.G = G
        self.D = D
        self.opt_D = opt_D
        self.opt_G = opt_G
        self.data_loader = data_loader
        self.stage = stage
        self.conversion_depth = conversion_depth

        self.toRGB = nn.Conv2d(self.conversion_depth, 2, 1)
        self.fromRGB = nn.Conv2d(2, self.conversion_depth, 1)
        self.latent_space = Variable(torch.FloatTensor(settings.BATCH_SIZE, 512, 1, 1))

        if settings.CUDA:
            self.toRGB.cuda()
            self.fromRGB.cuda()
            self.latent_space = self.latent_space.cuda()

        self.opt_toRGB = torch.optim.Adam(self.toRGB.parameters(), lr=settings.LEARNING_RATE)
        self.opt_fromRGB = torch.optim.Adam(self.fromRGB.parameters(), lr=settings.LEARNING_RATE)

        self.update_state = 0

    def get_rgb_layers(self):
        return self.toRGB, self.fromRGB

    def steps(self, n):
        for batch in cyclic_data_iterator(self.data_loader, n):
            print("Yo")
            batch = Variable(batch)
            if settings.CUDA:
                batch = batch.cuda()
            self.latent_space.data.normal_()
            fake = self.toRGB(self.G(self.latent_space))
            pred = self.D(self.fromRGB(fake))

            # Update G
            if self.update_state == settings.DISCRIMINATOR_ITERATIONS:
                self.update_state = 0
                loss = - torch.mean(pred)

                self.opt_G.zero_grad()
                self.opt_toRGB.zero_grad()
                loss.backward()
                self.opt_G.step()
                self.opt_toRGB.step()

            else:
                self.update_state += 1
                loss = torch.mean(pred) - torch.mean(self.D(self.fromRGB(batch)))

                if settings.GRADIENT_PENALTY:
                    grads = torch.autograd.grad(pred, self.D.parameters(),
                                                create_graph=True, allow_unused=True)
                    grad_norm = 0
                    for grad in grads:
                        if grad is not None:
                            grad_norm += grad.pow(2).sum()
                    grad_norm.sqrt_()

                    grad_loss = (grad_norm - 1).pow(2)
                    loss += 10 * grad_loss

                self.opt_D.zero_grad()
                self.opt_fromRGB.zero_grad()
                loss.backward()
                self.opt_fromRGB.step()
                self.opt_D.step()



