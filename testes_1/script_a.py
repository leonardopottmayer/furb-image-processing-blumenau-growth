import rasterio
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import cv2
from matplotlib.animation import FuncAnimation
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

# Analyze difference over all consecutive pairs
def process_images_for_changes(image_files):
    changes = []
    for i in range(len(image_files) - 1):
        with rasterio.open(image_files[i]) as first, rasterio.open(image_files[i + 1]) as second:
            # Read the third band and normalize
            first_image = first.read(3).astype(np.float32)
            last_image = second.read(3).astype(np.float32)
            first_image = ((first_image - first_image.min()) / (first_image.max() - first_image.min()) * 255).astype(np.uint8)
            last_image = ((last_image - last_image.min()) / (last_image.max() - last_image.min()) * 255).astype(np.uint8)
            
            # Equalize and calculate differences
            first_image_eq = cv2.equalizeHist(first_image)
            last_image_eq = cv2.equalizeHist(last_image)
            diff = cv2.absdiff(last_image_eq, first_image_eq)
            _, mask = cv2.threshold(diff, 70, 255, cv2.THRESH_BINARY)
            
            # Overlay the change mask
            last_image_color = cv2.cvtColor(last_image_eq, cv2.COLOR_GRAY2RGB)
            red_overlay = np.zeros_like(last_image_color)
            red_overlay[:, :, 0] = mask
            overlay = cv2.addWeighted(last_image_color, 0.7, red_overlay, 0.3, 0)
            
            changes.append((overlay, f"{i + 1} - {i + 2}"))
    return changes

# Generate and save animation as a GIF
def save_changes_as_gif(changes, filename="growth_animation.gif"):
    frames = []
    for change, period in changes:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(change)
        ax.set_title(f"Changes between {period}")
        ax.axis('off')
        fig.canvas.draw()
        
        # Convert matplotlib figure to PIL Image
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(image)
        plt.close(fig)
    
    # Save frames as a GIF
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=500, loop=0)
    print(f"GIF saved as {filename}")

# Process images and save animation
changes = process_images_for_changes(image_files)
save_changes_as_gif(changes, "growth_animation.gif")
