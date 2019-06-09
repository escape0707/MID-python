# Import dependencies
import copy
import time

import torch
from torch import nn, optim

# Import datasets and dataloaders
from dataset import testloader, trainloader
# Import iteration times of the training & learning rate of the Adam optimizer
# Import trained model filepath for saving
from parameters import iteration_times, learning_rate, trained_model_filepath


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

# Model training procedure
def train_model(model, dataloaders, criterion, optimizer, num_epochs=iteration_times):

    since = time.time()
    # Array for loss history of training and validating
    loss_history = {'train': [], 'val': []}

    # Train model by loop over the dataset multiple times
    for epoch in range(num_epochs):
        if show_log:
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Set loss counter
            running_loss = 0.0

            # Get the inputs
            for noisy, original in dataloaders[phase]:
                # Send inputs to correct device
                noisy, original = noisy.to(device), original.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward and track history if only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(noisy)
                    loss = criterion(outputs, original)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.item() * noisy.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if show_log:
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            loss_history[phase].append(epoch_loss)
        # Print a empty line between Losses of Epoches
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, loss_history


# Detect and use GPU acceleration when possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

show_log = __name__ == '__main__'

# Assuming that we are on a CUDA machine, this should print a CUDA device:
if show_log:
    print(device)

# Create model instance
model_ft = CNNDAE()

# Use optimal device to train the model
model_ft.to(device)

# Print the network architecture for sanity check reason
if show_log:
    print(model_ft)

# Dataloaders containing trainloader and testloader for training and validating
dataloaders_dict = {'train': trainloader, 'val': testloader}

# Set the criterion (the loss function) to be Mean Squared Error loss
criterion = nn.MSELoss()

# Set the optimizer (the optimize method) to be Adam algorithm
optimizer = optim.Adam(model_ft.parameters(), lr=learning_rate)

model_ft, loss_history = train_model(model_ft, dataloaders_dict, criterion, optimizer, iteration_times)

def save_model(model=model_ft, path=trained_model_filepath):
    torch.save(model.state_dict(), path)

#%%
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    for x in ['train', 'val']:
        ax.plot(loss_history[x], label=x)
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')

    # # Put a nicer background color on the legend.
    # legend.get_frame().set_facecolor('C0')
    # plt.savefig('D:\\loss.png')
    plt.show()

    save_model()
