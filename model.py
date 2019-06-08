# Import dependencies
import time
import copy

import torch
from torch import nn, optim
from torchvision.utils import make_grid

# Import datasets and dataloaders
from dataset import testloader, trainloader
# Import iteration times of the training & learning rate of the Adam optimizer
from parameters import iteration_times, learning_rate


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

    # Train model by loop over the dataset multiple times
    for epoch in range(num_epochs):
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
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            loss_history[phase].append(epoch_loss)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
        # Print statistics
        print('[%d] training loss: %.10d' %
            (epoch + 1, training_loss))
        
        # Set loss counter for validating
        valid_loss = 0.0
        for i, data in enumerate(testloader):
            # Get the inputs
            noisy, original = data[0].to(device), data[1].to(device)

            # Forward
            outputs = net(noisy)
            loss = criterion(outputs, original)

            # Sum loss
            valid_loss += loss.item()
            
        # Print statistics
        print('[%d] validating loss: %.10d' %
            (epoch + 1, valid_loss))

    print('Finised training.')



# functions to show an image
import matplotlib.pyplot as plt
import numpy as np

def sample_show(images, labels):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 4))
    image = make_grid(images, nrow=batch_size)
    label = make_grid(labels, nrow=batch_size)
    image = image / 2 + 0.5     # unnormalize
    label = label / 2 + 0.5     # unnormalize
    npimage = image.numpy()
    nplabel = label.numpy()
    ax[0].imshow(np.transpose(npimage, (1, 2, 0)))
    ax[1].imshow(np.transpose(nplabel, (1, 2, 0)))
    plt.show()


#%%
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
print(images.shape)
sample_show(images, labels)


#%%
net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = torch.clamp(net(images.to(device)).cpu(), min=-1, max=+1)
        print(outputs.shape)
        sample_show(outputs, labels)
