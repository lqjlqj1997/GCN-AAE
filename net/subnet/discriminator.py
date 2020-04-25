from torch import nn

class Discriminator(nn.Module):

    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear   (latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear   (64 , 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear   (32 , 1),
            nn.Sigmoid  (),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity