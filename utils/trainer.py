"""
Class that holds the main train loop, for a specific part of the training progress
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings


from utils.utils import cyclic_data_iterator

from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import time
import random


class StageTrainer:
    def __init__(self, G, D, data_loader, stage=6, conversion_depth=16, downscale_factor=1):
        self.G = G
        self.D = D
        self.data_loader = data_loader
        self.stage = stage
        self.conversion_depth = conversion_depth
        self.downscale_factor = downscale_factor

        self.toRGB = nn.Conv2d(self.conversion_depth, 2, 1)
        self.fromRGB = nn.Conv2d(2, self.conversion_depth, 1)
        self.latent_space = Variable(torch.FloatTensor(settings.BATCH_SIZE, 128, 1, 1))
        self.latent_ref_point = Variable(torch.FloatTensor(16, 128, 1, 1), volatile=True)
        self.pred_real = Variable(torch.zeros(1))
        self.pred_fake = Variable(torch.zeros(1))

        if settings.CUDA:
            self.toRGB.cuda()
            self.fromRGB.cuda()
            self.latent_space = self.latent_space.cuda()
            self.latent_ref_point = self.latent_ref_point.cuda()
            self.pred_real = self.pred_real.cuda()
            self.pred_fake = self.pred_fake.cuda()

        params_G = [param for param in G.parameters() if param.requires_grad]
        self.opt_G = torch.optim.Adamax(params_G,
                                        lr=settings.LEARNING_RATE,
                                        betas=settings.BETAS
                                        )
        params_D = [param for param in D.parameters() if param.requires_grad]
        self.opt_D = torch.optim.Adamax(params_D,
                                        lr=settings.LEARNING_RATE,
                                        betas=settings.BETAS,
                                        )
        self.opt_toRGB = torch.optim.Adamax(self.toRGB.parameters(),
                                            lr=settings.LEARNING_RATE,
                                            betas=settings.BETAS
                                            )
        self.opt_fromRGB = torch.optim.Adamax(self.fromRGB.parameters(),
                                              lr=settings.LEARNING_RATE,
                                              betas=settings.BETAS
                                              )

        # Don't load optimizer state
        # Load optimizer states, except for during fade in (which is freeze in now)
        #if settings.WORKING_MODEL and not settings.FADE_IN:
        #    self.opt_G.load_state_dict(torch.load("working_model/optG.state"))
        #    self.opt_D.load_state_dict(torch.load("working_model/optD.state"))

        self.update_state = 0

    def get_rgb_layers(self):
        return self.toRGB, self.fromRGB

    def save_fake_reference_batch(self, point):
        if not settings.WORKING_MODEL:
            raise RuntimeError("Won't save reference batch without working model")
        torch.manual_seed(1337)
        self.latent_ref_point.data.normal_()
        torch.manual_seed(int(time.clock()*1e6))
        fake = self.generate_fake(self.latent_ref_point)
        batch_shape = list(fake.shape)
        batch_shape[1] = 1
        single = make_grid(fake[:, 0].data.cpu().contiguous().view(batch_shape))
        save_image(single, "working_model/timelapse/fake_batch{}.png".format(point))

    def visualize(self, visualizer):
        # Get single batch using for loop
        for batch in cyclic_data_iterator(self.data_loader, 1):
            batch = Variable(batch, volatile=True)  # No backprop here
            if settings.CUDA:
                batch = batch.cuda()
            batch = F.avg_pool2d(batch, self.downscale_factor, stride=self.downscale_factor)
            self.latent_space.data.normal_()
            fake = self.generate_fake(Variable(self.latent_space.data, volatile=True))

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

            visualizer.update_loss(self.pred_real.data.cpu(), self.pred_fake.data.cpu())

    def steps(self, n):
        pred_real = self.pred_real  # This may not always be updated
        for i, batch in enumerate(cyclic_data_iterator(self.data_loader, n)):
            batch = Variable(batch)
            if settings.CUDA:
                batch = batch.cuda()
            batch = F.avg_pool2d(batch, self.downscale_factor, stride=self.downscale_factor)
            self.latent_space.data.normal_()
            fake = self.generate_fake(self.latent_space)
            pred_fake = self.predict(fake)

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
                pred_real = self.predict(batch)
                loss_D = pred_fake - pred_real

                if settings.GRADIENT_PENALTY:
                    params = [param for param in self.D.parameters() if param.requires_grad]
                    grads = torch.autograd.grad(torch.mean(pred_fake), params,
                                                create_graph=True, allow_unused=True)
                    grad_norm = 0
                    for grad in grads:
                        if grad is not None:
                            grad_norm += grad.pow(2).sum()
                    grad_norm.sqrt_()

                    grad_loss = (grad_norm - 1).pow(2)
                    #grad_loss = (grad_norm - 750).pow(2) / 562500
                    loss_D += grad_loss

                loss_D += 0.001 * pred_real.pow(2)  # Drift loss

                self.opt_D.zero_grad()
                self.opt_fromRGB.zero_grad()
                loss_D.backward()
                self.opt_fromRGB.step()
                self.opt_D.step()

            self.pred_real = pred_real * 0.1 + 0.9 * self.pred_real
            self.pred_fake = pred_fake * 0.1 + 0.9 * self.pred_fake

            if i % 10 == 9:
                print("Iter {}/{}     ".format(i+1, n), end="\r")
            self.update_hook()

    def generate_fake(self, latent_vector):
        return self.toRGB(self.G(latent_vector, levels=self.stage))

    def predict(self, image):
        return torch.mean(self.D(self.fromRGB(image), levels=self.stage))

    def update_hook(self):
        pass


class StochasticFadeInTrainer(StageTrainer):
    def __init__(self, G, D, data_loader, stage=6,
                 conversion_depth=32, downscale_factor=2,
                 next_cd=16, increment=0.01):
        super().__init__(G, D, data_loader, stage, conversion_depth, downscale_factor)

        self.next_cd = next_cd

        self.next_toRGB = nn.Conv2d(self.next_cd, 2, 1)
        self.next_fromRGB = nn.Conv2d(2, self.next_cd, 1)

        self.alpha = 0
        self.increment = increment

        if settings.CUDA:
            self.next_toRGB.cuda()
            self.next_fromRGB.cuda()

    def increment_alpha(self):
        self.alpha += self.increment
        self.alpha = min(self.alpha, 1)

    def generate_fake(self, latent_vector):
        if random.random() < self.alpha:
            big = self.next_toRGB(self.G(latent_vector, levels=self.stage + 1))
        else:
            big = F.upsample(self.toRGB(self.G(latent_vector, levels=self.stage)), scale_factor=2)
        return big

    def predict(self, image):
        if random.random() < self.alpha:
            pred = torch.mean(self.D(self.next_fromRGB(image), levels=self.stage+1))
        else:
            pred = torch.mean(self.D(self.fromRGB(F.avg_pool2d(image, 2, stride=2)), levels=self.stage))
        return pred

    def update_hook(self):
        self.increment_alpha()

    def get_rgb_layers(self):
        return (*super().get_rgb_layers(), self.next_toRGB, self.next_fromRGB)


class FadeInTrainer(StageTrainer):
    def __init__(self, G, D, data_loader, stage=6,
                 conversion_depth=32, downscale_factor=2,
                 next_cd=16, increment=0.01):
        super().__init__(G, D, data_loader, stage, conversion_depth, downscale_factor)

        self.next_cd = next_cd

        self.next_toRGB = nn.Conv2d(self.next_cd, 2, 1)
        self.next_fromRGB = nn.Conv2d(2, self.next_cd, 1)

        self.alpha = 0
        self.increment = increment

        if settings.CUDA:
            self.next_toRGB.cuda()
            self.next_fromRGB.cuda()

    def increment_alpha(self):
        self.alpha += self.increment
        self.alpha = min(self.alpha, 1)

    def generate_fake(self, latent_vector):
        big, small = self.G.fade_in(latent_vector, levels=self.stage+1)
        small = F.upsample(self.toRGB(small), scale_factor=2)
        big = self.next_toRGB(big)
        return (1 - self.alpha) * small + self.alpha * big

    def predict(self, image):
        small = F.avg_pool2d(image, 2, stride=2)
        big = self.next_fromRGB(image)
        small = self.fromRGB(small)
        return torch.mean(self.D.fade_in(big, small, self.alpha, levels=self.stage+1))

    def update_hook(self):
        self.increment_alpha()

    def get_rgb_layers(self):
        return (*super().get_rgb_layers(), self.next_toRGB, self.next_fromRGB)


# Ugly code for concept test, needs to be refactored if it gives good results
class FadeInLossTrainer(FadeInTrainer):
    def __init__(self, G, D, data_loader, stage=6,
                 conversion_depth=32, downscale_factor=2,
                 next_cd=16, increment=0.01):
        super().__init__(G, D, data_loader, stage,
                         conversion_depth, downscale_factor,
                         next_cd, increment)

    def generate_fake(self, latent_vector):
        big, small = self.generate_fakes(latent_vector)
        return (1 - self.alpha) * small + self.alpha * big

    def generate_fakes(self, latent_vector):
        big, small = self.G.fade_in(latent_vector, levels=self.stage+1)
        small = F.upsample(self.toRGB(small), scale_factor=2)
        big = self.next_toRGB(big)
        return big, small

    def predict(self, image):
        return torch.mean(self.D(self.next_fromRGB(image), levels=self.stage+1))

    def steps(self, n):
        pred_real = self.pred_real  # This may not always be updated
        for i, batch in enumerate(cyclic_data_iterator(self.data_loader, n)):
            batch = Variable(batch)
            if settings.CUDA:
                batch = batch.cuda()
            batch = F.avg_pool2d(batch, self.downscale_factor, stride=self.downscale_factor)
            self.latent_space.data.normal_()
            big, small = self.generate_fakes(self.latent_space)
            fake = (1 - self.alpha) * small + self.alpha * big
            pred_fake = self.predict(fake)

            # Update G
            if self.update_state == settings.DISCRIMINATOR_ITERATIONS:
                self.update_state = 0
                loss_G = - pred_fake + \
                         torch.mean(torch.abs(big - small.detach())) * (1 - self.alpha)

                self.opt_G.zero_grad()
                self.opt_toRGB.zero_grad()
                loss_G.backward()
                self.opt_G.step()
                self.opt_toRGB.step()

            else:
                self.update_state += 1
                pred_real = self.predict(batch)
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
                    #grad_loss = (grad_norm - 750).pow(2) / 562500
                    loss_D += 10 * grad_loss
                    loss_D += 0.0001 * pred_real.pow(2)

                self.opt_D.zero_grad()
                self.opt_fromRGB.zero_grad()
                loss_D.backward()
                self.opt_fromRGB.step()
                self.opt_D.step()

            self.pred_real = pred_real * 0.1 + 0.9 * self.pred_real
            self.pred_fake = pred_fake * 0.1 + 0.9 * self.pred_fake

            if i % 10 == 9:
                print("Iter {}/{}     ".format(i+1, n), end="\r")
            self.update_hook()

