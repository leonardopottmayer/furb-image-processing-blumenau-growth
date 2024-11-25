import rasterio
import matplotlib.pyplot as plt
import re
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Control variables
image_band = 3
enable_bnu = True

if enable_bnu:
    print('Blumenau')
    IMAGE_SIZE = (41, 125)
    pattern = r"(\d{1,2})_(\d{4})_BNU_250\.tif"
    images_path = r'./raster_bnu/'
else:
    print('SC')
    IMAGE_SIZE = (128, 87)
    pattern = r"(\d{1,2})_(\d{4})_UTM\.tif"
    images_path = r'./raster/'
    
# List all images in the specified path that match the pattern
images_files = [f for f in os.listdir(images_path) if re.match(pattern, f)]

# Sort images by date (month and year extracted from the file name)
images_files = sorted(
    [os.path.join(images_path, file) for file in os.listdir(images_path) if file.endswith('.tif')],
    key=lambda x: (int(re.search(pattern, x).group(2)), int(re.search(pattern, x).group(1)))
)

# Display the sorted list of images found
print(f"Found {len(images_files)} images in chronological order:")
for image_file in images_files:
    print(image_file)

# Original code to display each image with a label of month/year
fig, axes = plt.subplots(8, 15, figsize=(16,9))
axes = axes.flatten()

for index, file in enumerate(images_files):
    match = re.search(pattern, file)
    month = match.group(1)
    year = match.group(2)

    with rasterio.open(file) as src:
        img = src.read(image_band)
        ax = axes[index]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{month}/{year}")
        ax.axis('off')

plt.tight_layout()
plt.show()

# Calculate and display the average image for each year
# Group images by year
yearly_images = {}
for file in images_files:
    match = re.search(pattern, file)
    year = match.group(2)
    
    if year not in yearly_images:
        yearly_images[year] = []
    
    with rasterio.open(file) as src:
        img = src.read(image_band)
        yearly_images[year].append(img)

# Calculate the average image for each year and prepare for the GIF
average_images = []
for year, images in yearly_images.items():
    # Stack images and compute the mean across the stack
    mean_image = np.mean(images, axis=0)
    
    # Convert mean image to PIL format and add the year label
    pil_image = Image.fromarray((mean_image / mean_image.max() * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    
    # Optional: Set a font size and style
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Add year label at the top center of the image
    text = f"Year: {year}"
    text_width, text_height = draw.textsize(text, font=font)
    text_position = ((pil_image.width - text_width) // 2, 10)
    draw.text(text_position, text, fill="white", font=font)
    
    average_images.append(pil_image)

# Save the images as a GIF
gif_path = './yearly_evolution.gif'
average_images[0].save(gif_path, save_all=True, append_images=average_images[1:], duration=1000, loop=0)

print(f"GIF saved as {gif_path}")

# Generate a GIF showing only the change between the first and last year

# Retrieve the first and last years in chronological order
first_year = min(yearly_images.keys())
last_year = max(yearly_images.keys())

# Calculate the average images for the first and last years if not already done
first_year_image = np.mean(yearly_images[first_year], axis=0)
last_year_image = np.mean(yearly_images[last_year], axis=0)

# Convert to PIL format and add year labels
first_image_pil = Image.fromarray((first_year_image / first_year_image.max() * 255).astype(np.uint8)).convert("RGB")
last_image_pil = Image.fromarray((last_year_image / last_year_image.max() * 255).astype(np.uint8)).convert("RGB")
draw_first = ImageDraw.Draw(first_image_pil)
draw_last = ImageDraw.Draw(last_image_pil)

# Add labels for the first and last years
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

# Position the year label in the center top of the image
first_text = f"Year: {first_year}"
last_text = f"Year: {last_year}"
first_text_position = ((first_image_pil.width - draw_first.textsize(first_text, font=font)[0]) // 2, 10)
last_text_position = ((last_image_pil.width - draw_last.textsize(last_text, font=font)[0]) // 2, 10)

draw_first.text(first_text_position, first_text, fill="white", font=font)
draw_last.text(last_text_position, last_text, fill="white", font=font)

# Create the GIF with only the first and last images
gif_path_change = './change_first_last_year.gif'
first_image_pil.save(gif_path_change, save_all=True, append_images=[last_image_pil], duration=1000, loop=0)

print(f"GIF showing change from first to last year saved as {gif_path_change}")
