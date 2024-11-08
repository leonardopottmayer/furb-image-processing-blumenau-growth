import rasterio
import cv2
import numpy as np
import os
from glob import glob

from functions import show_image  # Função para mostrar imagens (conforme seu exemplo)

# Caminho da pasta com os arquivos raster
folder_path = './raster_bnu/'

# Listar todos os arquivos .tif na pasta e ordenar por data (assumindo que o nome inclui a data)
file_paths = sorted(glob(os.path.join(folder_path, '*.tif')))

# Inicializar uma variável para a imagem anterior (para calcular a diferença)
previous_img = None

# Loop através dos arquivos raster em ordem
for file_path in file_paths:
    with rasterio.open(file_path) as src:
        num_bands = src.count
        print(f"Processing {file_path} with {num_bands} bands")
        
        # Carregar e normalizar todas as bandas da imagem atual
        img_bands = []
        for band in range(1, num_bands + 1):
            img = src.read(band).astype(np.float32)
            img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_bands.append(img_normalized)

        # Converter a lista de bandas para uma imagem média em escala de cinza
        img_current = np.mean(img_bands, axis=0).astype(np.uint8)

        # Se já temos uma imagem anterior, calcular a diferença
        if previous_img is not None:
            # Calcular a diferença absoluta entre a imagem atual e a anterior
            diff_img = cv2.absdiff(img_current, previous_img)
            
            # Equalização do histograma para destacar as mudanças
            diff_equalized = cv2.equalizeHist(diff_img)
            
            # Binarização para destacar áreas com mudanças significativas
            _, diff_binary = cv2.threshold(diff_equalized, 50, 255, cv2.THRESH_BINARY)

            # Mostrar a imagem de diferença binarizada
            show_image(f"Diferença detectada em {os.path.basename(file_path)}", diff_binary)

            # Calcular o número de pixels alterados (área de mudança)
            changed_pixels = np.sum(diff_binary == 255)
            print(f"Área de mudança em pixels: {changed_pixels}")
        
        # Atualizar a imagem anterior para a próxima iteração
        previous_img = img_current
