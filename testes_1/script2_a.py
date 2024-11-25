import rasterio
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import cv2
from PIL import Image

# Control variables
ativa_bnu = False

if ativa_bnu:
    print('Blumenau')
    IMAGE_SIZE = (41, 125)
    pattern = r"(\d{1,2})_(\d{4})_BNU_250\.tif"
    images_path = r'./raster_bnu/'
else:
    print('SC')
    IMAGE_SIZE = (128, 87)
    pattern = r"(\d{1,2})_(\d{4})_UTM\.tif"
    images_path = r'./raster/'

# List of all images in sorted chronological order
image_files = sorted(
    [os.path.join(images_path, file) for file in os.listdir(images_path) if file.endswith('.tif')],
    key=lambda x: (int(re.search(pattern, x).group(2)), int(re.search(pattern, x).group(1)))
)

# Function to calculate the annual mean image
def calculate_annual_mean_images(image_files):
    annual_means = {}
    
    # Group images by year
    images_by_year = {}
    for file in image_files:
        match = re.search(pattern, file)
        month, year = int(match.group(1)), int(match.group(2))
        if year not in images_by_year:
            images_by_year[year] = []
        images_by_year[year].append(file)
    
    # Calculate mean for each year
    for year, files in images_by_year.items():
        sum_image = None
        for file in files:
            with rasterio.open(file) as src:
                image = src.read(1).astype(np.float32)  # Read band 3 and convert to float32 for averaging
                if sum_image is None:
                    sum_image = image
                else:
                    sum_image += image
        mean_image = (sum_image / len(files)).astype(np.uint8)  # Calculate the mean and convert to uint8
        annual_means[year] = cv2.equalizeHist(mean_image)  # Equalize histogram for visibility
        
    return annual_means

# Generate and save annual mean images as a GIF
def save_annual_means_as_gif(annual_means, filename="annual_growth_animation.gif"):
    frames = []
    for year, mean_image in sorted(annual_means.items()):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(mean_image, cmap='gray')
        ax.set_title(f"Year {year}")
        ax.axis('off')
        fig.canvas.draw()
        
        # Convert matplotlib figure to PIL Image
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(image)
        plt.close(fig)
    
    # Save frames as a GIF
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=800, loop=0)
    print(f"GIF saved as {filename}")

# Process images, calculate annual means, and save animation
annual_means = calculate_annual_mean_images(image_files)
save_annual_means_as_gif(annual_means, "annual_growth_animation.gif")
