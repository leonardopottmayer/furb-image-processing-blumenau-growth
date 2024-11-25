import rasterio
import cv2
import numpy as np

from functions import show_image

# Raster file path
file_path = './raster_bnu/10_2023_BNU_250.tif'

# Read the raster file and convert each band to an array (for OpenCV compatibility).
with rasterio.open(file_path) as src:
    amount_of_bands = src.count
    print(f"Amount of bands: {amount_of_bands}")
    
    # Loop through each band
    for band in range(1, amount_of_bands + 1):
        # Read the band and convert it to a float32 NumPy array (for OpenCV compatibility).
        img = src.read(band).astype(np.float32)
        
        # Normalizar a imagem para a faixa 0-255 (escala de cinza de 8 bits)
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Mostrar a imagem com matplotlib
        show_image(f"Banda {band}", img_normalized)
        
        # Exemplo de uso do OpenCV para redimensionar a imagem
        img_resized = cv2.resize(img_normalized, (100, 100))  # Redimensiona para 100x100 pixels
        
        # Exibir a imagem redimensionada
        show_image(f"Banda {band} Redimensionada", img_resized)
