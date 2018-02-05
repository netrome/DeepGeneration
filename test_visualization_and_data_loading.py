import utils.datasets as data
import visualizer
import torch


dataset = data.SyntheticFullyAnnotated("~/Data/DeepGeneration1")
visualizer = visualizer.Visualizer()

data_point = dataset[182]
visualizer.update_image(data_point[0], "real_img")
visualizer.update_image(data_point[1], "real_map")
visualizer.update_image(data_point.mean(0), "real_cat")

a = torch.randn(1)
b = torch.randn(1)
for i in range(100):
    a += torch.randn(1)
    b += torch.randn(1)
    visualizer.update_loss(a, b)

