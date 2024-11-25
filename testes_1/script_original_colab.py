# Imports
import rasterio
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import cv2
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from PIL import Image

# Control variables
ativa_bnu = False

if ativa_bnu:
  print('Blumenau')
  IMAGE_SIZE = (41, 125)
  pattern = r"(\d{1,2})_(\d{4})_BNU_250\.tif"
  file_path = './raster_bnu/10_2023_BNU_250.tif'
  images_path = r'./raster_bnu/'
else:
  print('SC')
  IMAGE_SIZE = (128, 87)
  pattern = r"(\d{1,2})_(\d{4})_UTM\.tif"
  file_path = './raster/10_2023_UTM.tif'
  images_path = r'./raster/'

image_files = sorted(
      [os.path.join(images_path, file) for file in os.listdir(images_path) if file.endswith('.tif')],
      key=lambda x: (int(re.search(pattern, x).group(2)), int(re.search(pattern, x).group(1)))
  )

images_count = len(image_files)
print(f'Quantidade de imagens: {images_count}')

with rasterio.open(file_path) as src:
    width, height = src.width, src.height

print(f'{width}, {height}')
print(f'Primeiro arquivo: {image_files[0]}')
print(f'Último arquivo: {image_files[-1]}')

# Bands visualization
fig, axes = plt.subplots(8, 15, figsize=(16,9))
axes = axes.flatten()

for index, file in enumerate(image_files):
    # print(file)
    match = re.search(pattern, file)
    month = match.group(1)
    year = match.group(2)
    # print(f"Arquivo: {file}, Mês: {month}, Ano: {year}")
    with rasterio.open(file) as src:
        img = src.read(1)
        ax = axes[index]
        ax.imshow(img)
        ax.set_title(f"{month}/{year}")
        ax.axis('off')

plt.tight_layout()
plt.show()

with rasterio.open(file_path) as src:
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))

    axes = axes.flatten()

    num_bands = src.count
    print(f"Número de bandas: {num_bands}")

    for band in range(1, num_bands + 1):
        img = src.read(band)

        ax = axes[band - 1]
        ax.imshow(img)
        ax.set_title(f"Banda {band}")
        ax.axis('off')

    for i in range(num_bands, 8):
      fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    
# CV2
with rasterio.open(image_files[0]) as first, rasterio.open(image_files[-1]) as second:
    first_image = first.read(3)
    last_image = second.read(3)

# Passo 1: precisa convertar para uint8, para conseguir equalizar
first_image = ((first_image - first_image.min()) / (first_image.max() - first_image.min()) * 255).astype(np.uint8)
last_image = ((last_image - last_image.min()) / (last_image.max() - last_image.min()) * 255).astype(np.uint8)

# Passo 2: Imagens equalizadas
first_image_equalized = cv2.equalizeHist(first_image)
last_image_equalized = cv2.equalizeHist(last_image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Primeira Imagem")
plt.imshow(first_image_equalized)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Última Imagem")
plt.imshow(last_image_equalized)
plt.axis('off')
plt.show()

# Passo 3: Diferença entre as imagens
difference_image = cv2.absdiff(last_image_equalized, first_image_equalized)

# Passo 4: Limita a diferença para destacar mudanças significativas
_, change_mask = cv2.threshold(difference_image, 70, 255, cv2.THRESH_BINARY)

# Passo 5: Gerar a imagem de diferença
last_image_color = cv2.cvtColor(last_image_equalized, cv2.COLOR_GRAY2RGB)
red_overlay = np.zeros_like(last_image_color)
red_overlay[:, :, 0] = change_mask
change_overlay = cv2.addWeighted(last_image_color, 0.7, red_overlay, 0.3, 0)

plt.figure(figsize=(10, 10))
plt.title("Primeira vs Última")
plt.imshow(change_overlay)
plt.axis('off')
plt.show()