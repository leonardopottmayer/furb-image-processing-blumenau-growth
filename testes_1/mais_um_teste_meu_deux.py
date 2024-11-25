import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Defina a pasta e o padrão de nome conforme a escolha do usuário
USE_RASTER_BNU = False  # Defina True para usar 'raster_bnu' ou False para usar 'raster'

# Configurações de acordo com a escolha da pasta
if USE_RASTER_BNU:
    folder_path = "./raster_bnu"
    file_pattern = re.compile(r'(\d+)_(\d+)_BNU_250\.tif')
else:
    folder_path = "./raster"
    file_pattern = re.compile(r'(\d+)_(\d+)_UTM\.tif')

# Função para processar cada imagem e obter as áreas iluminadas
def process_image(image_path):
    with rasterio.open(image_path) as src:
        img = src.read(1)
    
    # Converte a imagem para o tipo float32 antes de normalizar
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)  # Normaliza para 8 bits
    img = img.astype(np.uint8)  # Converte para uint8 após a normalização
    _, img_threshold = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)  # Define áreas iluminadas
    return img_threshold

# Função para comparar duas imagens e identificar crescimento
def detect_changes(image1, image2):
    diff = cv2.absdiff(image1, image2)
    _, diff_threshold = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    return diff_threshold

# Carregar e ordenar cronologicamente as imagens na pasta escolhida
def load_images(interval=12, path=folder_path):
    # Buscar todos os arquivos na pasta que correspondem ao padrão
    files = []

    # Ler todos os arquivos no diretório que combinam com o padrão
    for file_name in os.listdir(path):
        match = file_pattern.match(file_name)
        if match:
            month, year = int(match.group(1)), int(match.group(2))
            file_path = os.path.join(path, file_name)
            files.append((year, month, file_path))

    # Ordenar os arquivos por ano e mês
    files.sort()  # Ordena primeiro pelo ano, depois pelo mês
    images = [process_image(file[2]) for i, file in enumerate(files) if i % interval == 0]  # Seleciona em intervalos
    
    return images

# Visualizar o crescimento urbano ao longo do tempo
def visualize_growth(images):
    fig, axs = plt.subplots(1, len(images) - 1, figsize=(15, 5))
    for i in range(1, len(images)):
        growth = detect_changes(images[i - 1], images[i])
        axs[i - 1].imshow(growth, cmap='hot', interpolation='nearest')
        axs[i - 1].set_title(f"Ano {2014 + i - 1}")
        axs[i - 1].axis('off')
    plt.show()

# Carregar imagens, analisar crescimento e exibir visualização
images = load_images(interval=12)  # Carregar uma imagem por ano (10 anos)
visualize_growth(images)
