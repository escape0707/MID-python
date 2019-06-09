from os.path import join

# Training parameters
iteration_times = 100
batch_size = 10
learning_rate = 0.001


# Directory paths
project_directory = "D:\\Digital_Media_Technology\\Workspaces\\Python\\MID-python"
original_images_directory = join(project_directory,"original_images")
processed_images_directory = join(project_directory,"images")
trained_model_filepath = join(project_directory, "trained_model.pt")


# Processed image size
processed_image_size = 64, 64


# Noise mode default to Gaussian
noise_mode = 'Gaussian'

# Mean (mu) and standard deviation (sigma) for noise signal generation
mu, sigma = 0, 0.05
