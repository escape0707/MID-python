import os

from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Import batch size & data file directory
from parameters import batch_size, processed_images_directory


# Find and index image files
def index_data():

    # Clear image indices
    noisy_images.clear()
    original_images.clear()

    # Lookup noisy images
    for dirpath, dirnames, filenames in os.walk(os.path.join(processed_images_directory, 'noisy')):
        for filename in filenames:
            noisy_images.append(os.path.join(dirpath, filename))

    # Lookup original images
    for dirpath, dirnames, filenames in os.walk(os.path.join(processed_images_directory, 'original')):
        for filename in filenames:
            original_images.append(os.path.join(dirpath, filename))


# Define training dataset
class DenoiseDataSet(Dataset):

    def __init__(self, noisy_images, original_images, transform=None):
        self.noisy_images = noisy_images
        self.original_images = original_images
        self.transform = transform

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, index):

        # Select noisy & original filepath
        noisy_filepath = self.noisy_images[index]
        original_filepath = self.original_images[index]

        # Read noisy & original image
        noisy = io.imread(noisy_filepath)
        original = io.imread(original_filepath)

        # Apply transform
        if self.transform:
            noisy = self.transform(noisy)
            original = self.transform(original)

        # Return noisy & original image as requested
        return noisy, original


# Image filepath indices list
noisy_images = []
original_images = []

# Initialize file indices of training set & testing set
index_data()

# Transform which convert the input into a Tensor and normalize its content into a range of [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# Initialize datasets
trainset = DenoiseDataSet(noisy_images[:300] + noisy_images[-300:],
                          original_images[:300] + original_images[-300:],
                          transform)
testset = DenoiseDataSet(noisy_images[300:-300],
                         original_images[300:-300],
                         transform)

# Initialize DataLoader
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size)

#%%
# Simple tests to show some noisy & original images and other infos from both DataLoader
if __name__ == "__main__":
#%%
    # Check the filepath of two image pairs
    print(noisy_images[0], noisy_images[-1])
    print(original_images[0], original_images[-1])
#%%
    # Check the filepath of the first ten image pairs
    for i in range(10):
        print(noisy_images[i])
        print(original_images[i])
#%%
    # Check the total of images
    print(len(noisy_images), len(original_images))
#%%
    # Import modules for images displaying
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision.utils import make_grid

    # Function to show an tensor in specified axes
    def tensor_show(ax, tensor):
        npimg = tensor.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))

    # Generate iterater on dataloader
    trainiter = iter(trainloader)
    testiter = iter(testloader)
#%%
    # %matplotlib
    # Get images from iteraters
    data_mini_batches = trainiter.next() + testiter.next()

    # Plot first four images of the mini-batch from all dataloaders
    fig, axs = plt.subplots(4, sharex=True, sharey=True)
    for i in range(4):
        tensor_show(axs[i], make_grid(data_mini_batches[i][:4], nrow=4, normalize=True, range=(-1, 1)))

    plt.show()