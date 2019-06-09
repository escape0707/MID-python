import numpy as np

# Function to show an tensor in specified axes
def tensor_show(ax, tensor):
    ndarray = tensor.numpy()
    ndarray_show(ax, ndarray)

def ndarray_show(ax, ndarray):
    ax.imshow(np.transpose(ndarray, (1, 2, 0)))