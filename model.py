from torch import nn, optim

# Define convolutional encoding block architecture
def encode_block(in_f, out_f):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, 5, padding=2),
        nn.MaxPool2d(2),
        nn.ReLU()
    )

# Define convolutional decoding block architecture
def decode_block(in_f, out_f):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, 5, padding=2),
        nn.ReLU(),
        nn.Upsample(scale_factor=2)
    )

# Define convolutional auto encode-decode neural network architecture
class CNNDAE(nn.Module):
    def __init__(self):
        super(CNNDAE, self).__init__()

        self.encoder = nn.Sequential(
            encode_block(1, 64),
            encode_block(64, 64)
        )

        self.decoder = nn.Sequential(
            decode_block(64, 64),
            decode_block(64, 64),
            nn.Conv2d(64, 1, 5, padding=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x