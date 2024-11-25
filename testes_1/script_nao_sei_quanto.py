import rasterio
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import cv2
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from PIL import Image
import tensorflow as tf

ativa_bnu = False

if ativa_bnu:
  print('Blumenau')
  IMAGE_SIZE = (41, 125)
  pattern = r"(\d{1,2})_(\d{4})_BNU_250\.tif"
  file_path = '/content/drive/MyDrive/[03] Faculdade/[04] Conteudo/image-processing/images/Blumenau/10_2023_BNU_250.tif'
  images_path = r'/content/drive/MyDrive/[03] Faculdade/[04] Conteudo/image-processing/images/Blumenau/'
else:
  print('SC')
  IMAGE_SIZE = (128, 87)
  pattern = r"(\d{1,2})_(\d{4})_UTM\.tif"
  file_path = '/content/drive/MyDrive/[03] Faculdade/[04] Conteudo/image-processing/images/Santa Catarina/10_2023_UTM.tif'
  images_path = r'/content/drive/MyDrive/[03] Faculdade/[04] Conteudo/image-processing/images/Santa Catarina/'

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

with rasterio.open(image_files[0]) as first, rasterio.open(image_files[-1]) as second:
    first_image = first.read(1)
    last_image = second.read(1)

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
_, change_mask = cv2.threshold(difference_image, 50, 255, cv2.THRESH_BINARY)

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

# Função para converter e equalizar uma imagem usando técnicas mais próximas do OpenCV
def load_and_preprocess_image(file, band_number=1, img_size=(128, 128)):
    # Carregar a imagem em escala de cinza diretamente
    with rasterio.open(file) as src:
        img = src.read(band_number).astype(np.float32)

    # Redimensionamento com aproximação da qualidade OpenCV
    img_resized = tf.image.resize(img[..., tf.newaxis], img_size, method='lanczos3')

    # Normalizar para o intervalo 0-1 e reescalar para 0-255
    img_normalized = (img_resized - tf.reduce_min(img_resized)) / (tf.reduce_max(img_resized) - tf.reduce_min(img_resized)) * 255
    img_normalized = tf.cast(img_normalized, tf.uint8)

    # Ajuste de contraste para imitar equalização do OpenCV
    img_equalized = tf.image.adjust_contrast(tf.cast(img_normalized, tf.float32), contrast_factor=2)
    img_equalized = tf.squeeze(img_equalized)  # Remover dimensão extra

    return img_equalized

# Carregar e pré-processar as imagens
first_image_equalized = load_and_preprocess_image(image_files[0], img_size=(128, 128))
last_image_equalized = load_and_preprocess_image(image_files[-1], img_size=(128, 128))

# Exibir as imagens equalizadas para comparação
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Primeira Imagem Equalizada")
plt.imshow(first_image_equalized)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Última Imagem Equalizada")
plt.imshow(last_image_equalized)
plt.axis('off')
plt.show()

# Calcular a diferença entre as imagens apenas para aumentos de luminosidade
difference_image = tf.maximum(last_image_equalized - first_image_equalized, 0)

# Aplicar um limiar para destacar mudanças significativas
threshold_value = 50
change_mask = tf.where(difference_image > threshold_value, 255, 0)
change_mask = tf.cast(change_mask, tf.uint8)

# Criar a sobreposição vermelha para a área central
last_image_color = tf.stack([last_image_equalized, last_image_equalized, last_image_equalized], axis=-1)
last_image_color = tf.cast(last_image_color, tf.float32)  # Converter para float32 para operações de multiplicação com decimais

# Criar a máscara vermelha para sobreposição
red_overlay = tf.stack([change_mask, tf.zeros_like(change_mask), tf.zeros_like(change_mask)], axis=-1)
red_overlay = tf.cast(red_overlay, tf.float32)

# Combinar a imagem original com a máscara vermelha
change_overlay = tf.add(last_image_color * 0.7, red_overlay * 0.3)
change_overlay = tf.cast(change_overlay, tf.uint8)

# Exibir o resultado final
plt.figure(figsize=(10, 10))
plt.title("Primeira vs Última - Mudanças em Vermelho")
plt.imshow(change_overlay)
plt.axis('off')
plt.show()