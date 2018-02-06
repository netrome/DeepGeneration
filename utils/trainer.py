"""
Class that holds the main train loop, for a specific part of the training progress
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings


from utils.utils import cyclic_data_iterator

from torch.autograd import Variable


class StageTrainer:

    def __init__(self, G, D, opt_G, opt_D, data_loader, stage=6, conversion_depth=16, downscale_factor=1):
        self.G = G
        self.D = D
        self.opt_D = opt_D
        self.opt_G = opt_G
        self.data_loader = data_loader
        self.stage = stage
        self.conversion_depth = conversion_depth
        self.downscale_factor = downscale_factor

        self.toRGB = nn.Conv2d(self.conversion_depth, 2, 1)
        self.fromRGB = nn.Conv2d(2, self.conversion_depth, 1)
        self.latent_space = Variable(torch.FloatTensor(settings.BATCH_SIZE, 512, 1, 1))
        self.pred_real = Variable(torch.zeros(1))
        self.pred_fake = Variable(torch.zeros(1))

        if settings.CUDA:
            self.toRGB.cuda()
            self.fromRGB.cuda()
            self.latent_space = self.latent_space.cuda()
            self.pred_real = self.pred_real.cuda()
            self.pred_fake = self.pred_fake.cuda()

        self.opt_toRGB = torch.optim.Adam(self.toRGB.parameters(), lr=settings.LEARNING_RATE)
        self.opt_fromRGB = torch.optim.Adam(self.fromRGB.parameters(), lr=settings.LEARNING_RATE)

        self.update_state = 0

    def get_rgb_layers(self):
        return self.toRGB, self.fromRGB

    def visualize(self, visualizer):
        # Get single batch using for loop
        for batch in cyclic_data_iterator(self.data_loader, 1):
            batch = Variable(batch)
            if settings.CUDA:
                batch = batch.cuda()
            batch = F.max_pool2d(batch, self.downscale_factor, stride=self.downscale_factor)
            self.latent_space.data.normal_()
            fake = self.toRGB(self.G(self.latent_space, levels=self.stage))

            batch_shape = list(batch.shape)
            batch_shape[1] = 1

            visualizer.update_image(batch[0][0].data.cpu(), "real_img")
            visualizer.update_image(batch[0][1].data.cpu(), "real_map")
            visualizer.update_image(batch[0].mean(0).data.cpu(), "real_cat")
            visualizer.update_batch(
                batch[:, 0].data.cpu().contiguous().view(batch_shape), "real_batch")

            fake.data.clamp_(0, 1)
            visualizer.update_image(fake[0][0].data.cpu(), "fake_img")
            visualizer.update_image(fake[0][1].data.cpu(), "fake_map")
            visualizer.update_image(fake[0].mean(0).data.cpu(), "fake_cat")

            visualizer.update_batch(
                fake[:, 0].data.cpu().contiguous().view(batch_shape), "fake_batch")

            visualizer.update_loss(self.pred_fake.data.cpu(), self.pred_real.data.cpu())

    def steps(self, n):
        print("Training for {} iterations".format(n))
        pred_fake = self.pred_fake
        pred_real = self.pred_real
        for i, batch in enumerate(cyclic_data_iterator(self.data_loader, n)):
            batch = Variable(batch)
            if settings.CUDA:
                batch = batch.cuda()
            batch = F.max_pool2d(batch, self.downscale_factor, stride=self.downscale_factor)
            self.latent_space.data.normal_()
            fake = self.toRGB(self.G(self.latent_space, levels=self.stage))
            pred_fake = torch.mean(self.D(self.fromRGB(fake), levels=self.stage))

            # Update G
            if self.update_state == settings.DISCRIMINATOR_ITERATIONS:
                self.update_state = 0
                loss_G = - pred_fake

                self.opt_G.zero_grad()
                self.opt_toRGB.zero_grad()
                loss_G.backward()
                self.opt_G.step()
                self.opt_toRGB.step()

            else:
                self.update_state += 1
                pred_real = torch.mean(self.D(self.fromRGB(batch), levels=self.stage))
                loss_D = pred_fake - pred_real

                if settings.GRADIENT_PENALTY:
                    grads = torch.autograd.grad(torch.mean(pred_fake), self.D.parameters(),
                                                create_graph=True, allow_unused=True)
                    grad_norm = 0
                    for grad in grads:
                        if grad is not None:
                            grad_norm += grad.pow(2).sum()
                    grad_norm.sqrt_()

                    grad_loss = (grad_norm - 1).pow(2)
                    loss_D += 10 * grad_loss

                self.opt_D.zero_grad()
                self.opt_fromRGB.zero_grad()
                loss_D.backward()
                self.opt_fromRGB.step()
                self.opt_D.step()

            self.pred_real = pred_real * 0.1 + 0.9 * self.pred_real
            self.pred_fake = pred_fake * 0.1 + 0.9 * self.pred_fake

            print("Iter {}/{}     ".format(i, n), end="\r")



