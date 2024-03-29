from visdom import Visdom
import torch


class Visualizer:
    def __init__(self):
        self.vis = Visdom()

        self.real_image = None
        self.real_heatmap = None
        self.real_concat = None
        self.real_batch = None
        self.fake_image = None
        self.fake_concat = None
        self.fake_heatmap = None
        self.fake_batch = None

        self.loss = None

        self.point = 0

    def initiate_windows(self):
        random_image = torch.rand(1, 256, 256)
        random_batch = torch.rand(8, 1, 256, 256)
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
                "title": "Training progression",
                "legend": ["pred real", "pred fake"],
            }
        )
        self.point = 0

    def update_image(self, img, name):
        self.vis.image(img, win=name, opts={"caption": name})

    def update_batch(self, batch, name):
        self.vis.images(batch, win=name, opts={"caption": name}, nrow=8)

    def update_loss(self, pred_real, pred_fake):
        self.vis.line(
            X=torch.ones(1, 2)*self.point,
            Y=torch.stack([pred_real, pred_fake], dim=1),
            win="loss",
            update="append",
            opts={
                "xlabel": "Iteration",
                "ylabel": "Loss",
                "title": "Here is a title",
                "legend": ["pred real", "pred fake"],
            }
        )
        self.point += 1
