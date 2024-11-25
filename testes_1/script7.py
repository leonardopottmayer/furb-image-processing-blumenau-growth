import rasterio
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

# Caminho para as imagens .tif
imagens_path = './raster/*.tif'

# Função para ordenar as imagens com base no nome do arquivo (mês e ano)
def ordenar_imagens(imagens_path):
    imagens_ordenadas = sorted(
        glob.glob(imagens_path),
        key=lambda x: (
            int(re.search(r'_(\d{1,2})_(\d{4})_', x).group(2)) if re.search(r'_(\d{1,2})_(\d{4})_', x) else 0,  # Ano
            int(re.search(r'_(\d{1,2})_(\d{4})_', x).group(1)) if re.search(r'_(\d{1,2})_(\d{4})_', x) else 0   # Mês
        )
    )
    return imagens_ordenadas

# Carregar e organizar as imagens
imagens = []
for img_path in ordenar_imagens(imagens_path):
    with rasterio.open(img_path) as src:
        img = src.read(1)  # Lê a primeira banda
        imagens.append(img)

# Calcular a diferença entre a primeira e a última imagem para identificar o crescimento
primeira_imagem = imagens[0]
ultima_imagem = imagens[-1]
diferenca_crescimento = np.clip(ultima_imagem - primeira_imagem, 0, None)  # Somente valores positivos

# Plotar a imagem da diferença para observar o crescimento
plt.figure(figsize=(10, 10))
plt.title("Diferença de Brilho Noturno (Crescimento da Cidade)")
plt.imshow(diferenca_crescimento, cmap='hot')
plt.colorbar(label="Intensidade de Crescimento")
plt.show()
