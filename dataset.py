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
class TrainSet(Dataset):
    
    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        # Training set size is 600
        return 600
    
    def __getitem__(self, index):
        
        # Training samples size from dataset 1 and 2 is 300 and 300
        # Testing samples size from dataset 1 and 2 is 22 and 100
        if index < 300:
            img_name = noisy_images[index]
            label_name = original_images[index]
        else:
            img_name = noisy_images[index + 22]
            label_name = original_images[index + 22]
            
        # Read original and noisy image
        image = io.imread(img_name)
        label = io.imread(label_name)

        # Apply transform
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            
        # Return original and noisy image as requested
        return image, label


# Definition of testing dataset
class TestSet(Dataset):
    
    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        # Testing set size is 122
        return 122
    
    def __getitem__(self, index):
        
        # Training samples size from dataset 1 and 2 is 300 and 300
        # Testing samples size from dataset 1 and 2 is 22 and 100
        if index < 22:
            img_name = noisy_images[index + 300]
            label_name = original_images[index + 300]
        else:
            img_name = noisy_images[index + 600]
            label_name = original_images[index + 600]
            
        # Read original and noisy image
        image = io.imread(img_name)
        label = io.imread(label_name)

        
        # Apply transform
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            
        # Return original and noisy image as requested
        return image, label


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
