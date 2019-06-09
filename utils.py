import numpy as np

# Function to show an tensor in specified axes
def tensor_show(ax, tensor):
    npimg = tensor.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))