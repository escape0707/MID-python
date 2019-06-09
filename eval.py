#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import median
from skimage.measure import compare_ssim
from skimage.restoration import denoise_nl_means
from torchvision.utils import make_grid

from dataset import testloader
from model import CNNDAE
from parameters import device, trained_model_filepath
from utils import ndarray_show, tensor_show

n_rows, n_columns = 5, 3

#%%
model_ft = CNNDAE()
model_ft.load_state_dict(torch.load(trained_model_filepath))
model_ft.eval()
model_ft.to(device)

testiter = iter(testloader)

#%%
# Origin
# Noisy
# Non-local means
# Median filter
# CNNDAE
# SSIM
noisy, original = testiter.next()
images = [noisy[:n_columns], original[:n_columns]]

with torch.no_grad():
    outputs = model_ft(noisy[:n_columns].to(device)).to('cpu')

images.append(outputs)

for tensor in images:
    # Unnormalize tensor
    tensor += 1
    tensor /= 2

noisy_images = images[0].numpy()

non_local_means = []
median_filter = []
def img_to_tensor(img):
    return torch.unsqueeze(torch.from_numpy(img), 0)

for noisy_image in noisy_images:
    noisy_image = np.squeeze(noisy_image)
    non_local_means.append(img_to_tensor(denoise_nl_means(noisy_image)))
    median_filter.append(img_to_tensor(median(noisy_image)))

processed_images = images + [non_local_means, median_filter]
method_names = ['Noisy','Origin','CNNDAE','Non-local means','Median filter']
ssim = []
fig, axs = plt.subplots(n_rows, sharex=True, sharey=True)
for i, images in enumerate(processed_images):
    tensor_show(axs[i], make_grid(images))

    ssim.append([])
    for j, image in enumerate(images):
        ssim[i].append(compare_ssim(np.squeeze(original[j].numpy()), np.squeeze(image.numpy())))

    print('SSIM for %s is: %.6f' % (method_names[i], np.mean(ssim[i])))
plt.show()
