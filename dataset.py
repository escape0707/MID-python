import os

import torch
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
trainset = TrainSet(transform)
testset = TestSet(transform)

# Initialize DataLoader
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size)


if __name__ == "__main__":
    print(noisy_images[0], noisy_images[-1])
    print(original_images[0], original_images[-1])
    for i in range(10):
        print(noisy_images[i])
        print(original_images[i])
    print(len(noisy_images), len(original_images))


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision
    
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    imshow(torchvision.utils.make_grid(images))
