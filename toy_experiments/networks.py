import torch.nn as nn

hidden_size = 100

latent_size = 2

encoder = nn.Sequential(
        nn.Linear(3, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 2*latent_size)
        )

decoder = nn.Sequential(
        nn.Linear(latent_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 3)
        )

discriminator = nn.Sequential(
        nn.Linear(3, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)
        )

classifier = nn.Sequential(
        nn.Linear(2, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)
        )
