import shutil
import os

import torch
from torchvision.utils import save_image

from dataset import testloader
from model import CNNDAE
from parameters import device, final_output_directory, trained_model_filepath


if __name__ == "__main__":
    model_ft = CNNDAE()
    model_ft.load_state_dict(torch.load(trained_model_filepath))
    model_ft.eval()
    model_ft.to(device)

    # shutil.rmtree(final_output_directory, ignore_errors=True)
    os.makedirs(final_output_directory, exist_ok=True)

    with torch.no_grad():
        for i, (noisy, _) in enumerate(testloader):
            outputs = torch.clamp(model_ft(noisy.to(device)).cpu(), min=-1, max=1)
            for j, output in enumerate(outputs):
                final_filepath = os.path.join(final_output_directory, str(i * testloader.batch_size + j) + '.png')
                save_image(output, final_filepath, normalize=True, range=(-1, 1))
