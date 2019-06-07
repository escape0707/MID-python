import os
import shutil
import skimage
from skimage import io, img_as_float


# Import input and output images directories
from parameters import original_images_directory as input_directory
from parameters import processed_images_directory as output_directory

# Import output image size
from parameters import processed_image_size

# Import noise mode, mu & sigma for noise signal generation arguments
from parameters import noise_mode, mu, sigma


# Purge the output folder
def purge_output_folder():
    shutil.rmtree(output_directory, ignore_errors=True)
    os.makedirs(output_directory, exist_ok=True)


# Find every image in input directory
# Then read, resize, normalize, add noise and save processes
def process_data(input_directory=input_directory, 
                 output_directory=output_directory,
                 output_size=processed_image_size,
                 noise_mode=noise_mode,
                 mu=mu,
                 sigma=sigma,
                 save_original=True,
                 save_noisy=True
                 ):

    for folder_index, (dirpath, dirnames, filenames) in enumerate(os.walk(input_directory)):

        # Calculate and create output folders paths
        output_directory_original = os.path.join(output_directory, 'original', str(folder_index))
        output_directory_noisy = os.path.join(output_directory, 'noisy', str(folder_index))
        os.makedirs(output_directory_original, exist_ok=True)
        os.makedirs(output_directory_noisy, exist_ok=True)

        # Find and process image in each folder
        for file_index, filename in enumerate(filenames):

            # Calculate input/output filepath/filename
            infile_fullpath = os.path.join(dirpath, filename)
            outfile_name = str(file_index) + '.bmp'

            # Try read image
            try:
                image = img_as_float(io.imread(infile_fullpath, as_gray=True))
            except:
                continue

            # Resize to output_size and normalize
            image = skimage.transform.resize(image, output_size)
            image -= image.min()
            image /= image.max()

            if save_original:
                # Save resized original image
                outfile_original = os.path.join(output_directory_original, outfile_name)
                io.imsave(outfile_original, image)

            if save_noisy:
                # Add noise
                noisy = skimage.util.random_noise(image, mode=noise_mode, mean=mu, var=sigma**2)

                # Save noisy image
                outfile_noisy = os.path.join(output_directory_noisy, outfile_name)
                io.imsave(outfile_noisy, noisy)


if __name__ == "__main__":  
    purge_output_folder()
    process_data()
