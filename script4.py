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

# Inicializar uma variável para a média acumulada e a última imagem
average_img = None
last_img = None
num_images = 0

# Loop através dos arquivos raster em ordem para calcular a média acumulada
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

        # Atualizar a média acumulada
        if average_img is None:
            average_img = img_current.astype(np.float32)
        else:
            average_img += img_current
        num_images += 1

        # Salvar a última imagem para comparação posterior
        last_img = img_current

# Calcular a média final
average_img = (average_img / num_images).astype(np.uint8)

# Comparar a última imagem com a média acumulada
diff_img = cv2.absdiff(last_img, average_img)

# Destacar áreas onde houve um aumento de pixels claros
_, diff_binary = cv2.threshold(diff_img, 50, 255, cv2.THRESH_BINARY)

# Criar uma imagem colorida a partir da última imagem
colored_last_img = cv2.cvtColor(last_img, cv2.COLOR_GRAY2BGR)

# Definir a cor vermelha para áreas em crescimento
red = np.zeros_like(colored_last_img)
red[:, :] = [0, 0, 255]  # BGR para vermelho

# Aplicar a cor vermelha nas áreas onde a diferença é significativa
growth_mask = diff_binary == 255
colored_growth_img = np.where(growth_mask[:, :, None], red, colored_last_img)

# Mostrar a imagem com as áreas de crescimento em vermelho
show_image('Áreas de Crescimento em Vermelho', colored_growth_img)
