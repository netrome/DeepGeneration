import visualizer
import settings

from utils import datasets
from utils import progressive_networks
import utils.utils as utils

from torch.autograd import Variable
from torch.utils.data import DataLoader

# Get utilities ---------------------------------------------------
dataset = datasets.SyntheticFullyAnnotated(settings.DATA_PATH)
data_loader = DataLoader(dataset,
                         batch_size=settings.BATCH_SIZE,
                         shuffle=True,
                         pin_memory=True)
visualizer = visualizer.Visualizer()

# Define networks -------------------------------------------------
G = progressive_networks.TrivialPenerator()
D = progressive_networks.TrivialDiscriminator()

point = dataset[12]
print(point.shape)
down = utils.downsample_tensor(Variable(point), 1).data
print(down.shape)

visualizer.update_image(down[0], "real_img")
visualizer.update_image(down[1], "real_map")
visualizer.update_image(down.mean(0), "real_cat")



