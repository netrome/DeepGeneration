from visdom import Visdom
import torch

random_image = torch.rand(1, 256, 256)
random_batch = torch.rand(8, 1, 256, 256)


class Visualizer:
    def __init__(self):
        self.vis = Visdom()

        self.real_image = self.vis.image(random_image, win="real_img", opts={"caption": "Real image"})
        self.real_heatmap = self.vis.image(random_image, win="real_map")
        self.real_concat = self.vis.image(random_image, win="real_cat")
        self.real_batch = self.vis.images(random_batch, win="real_batch")
        self.fake_image = self.vis.image(random_image, win="fake_img")
        self.fake_concat = self.vis.image(random_image, win="fake_cat")
        self.fake_heatmap = self.vis.image(random_image, win="fake_map")
        self.fake_batch = self.vis.images(random_batch, win="fake_batch")

        self.loss = self.vis.line(
            X=torch.zeros(1),
            Y=torch.zeros(1, 2),
            win="loss",
            opts={
                "xlabel": "Iteration",
                "ylabel": "Loss",
                "title": "Here is a title",
                "legend": ["Loss G", "Loss D"],
            }
        )
        self.point = 0

    def update_image(self, img, name):
        self.vis.image(img, win=name, opts={"caption": name})

    def update_batch(self, batch, name):
        self.vis.images(batch, win=name, opts={"caption": name}, nrow=4)

    def update_loss(self, loss_G, loss_D):
        self.vis.line(
            X=torch.ones(1, 2)*self.point,
            Y=torch.stack([loss_G, loss_D], dim=1),
            win=self.loss,
            update="append",
            opts={
                "xlabel": "Iteration",
                "ylabel": "Loss",
                "title": "Here is a title",
                "legend": ["Loss G", "Loss D"],
            }
        )
        self.point += 1
