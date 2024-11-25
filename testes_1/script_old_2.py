from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import rasterio
import matplotlib.pyplot as plt
import re
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# Control variables
image_band = 3
enable_bnu = True

if enable_bnu:
    print('Blumenau')
    IMAGE_SIZE = (41, 125)
    pattern = r"(\d{1,2})_(\d{4})_BNU_250\.tif"
    images_path = r'./raster_bnu/'
    files_to_remove = ["7_2022_BNU_250.tif"]
else:
    print('SC')
    IMAGE_SIZE = (128, 87)
    pattern = r"(\d{1,2})_(\d{4})_UTM\.tif"
    images_path = r'./raster/'
    files_to_remove = ["07_2022_UTM.tif"]

# List all images in the specified path that match the pattern
images_files = [f for f in os.listdir(images_path) if re.match(pattern, f)]

# Remove the specified files if they exist
images_files = [f for f in images_files if f not in files_to_remove]

print("Remaining files:", images_files)

# Sort images by date (month and year extracted from the file name)
images_files = sorted(
    [os.path.join(images_path, file) for file in images_files if file.endswith('.tif')],
    key=lambda x: (int(re.search(pattern, x).group(2)), int(re.search(pattern, x).group(1)))
)

# Display the sorted list of images found
print(f"Found {len(images_files)} images in chronological order (after removal):")
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
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
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
first_text = f"Year: {first_year}"
last_text = f"Year: {last_year}"

first_text_bbox = draw_first.textbbox((0, 0), first_text, font=font)
last_text_bbox = draw_last.textbbox((0, 0), last_text, font=font)

first_text_width = first_text_bbox[2] - first_text_bbox[0]
first_text_height = first_text_bbox[3] - first_text_bbox[1]
last_text_width = last_text_bbox[2] - last_text_bbox[0]
last_text_height = last_text_bbox[3] - last_text_bbox[1]

first_text_position = ((first_image_pil.width - first_text_width) // 2, 10)
last_text_position = ((last_image_pil.width - last_text_width) // 2, 10)

draw_first.text(first_text_position, first_text, fill="white", font=font)
draw_last.text(last_text_position, last_text, fill="white", font=font)

# Create the GIF with only the first and last images
gif_path_change = './change_first_last_year.gif'
first_image_pil.save(gif_path_change, save_all=True, append_images=[last_image_pil], duration=1000, loop=0)

print(f"GIF showing change from first to last year saved as {gif_path_change}")

# -------------------------
# IMPLEMENTAR SVM
# -------------------------

# 1. Preparar os dados (extração de pixels e normalização)
pixel_data = []  # Lista para armazenar os valores dos pixels
labels = []      # Lista para armazenar os rótulos correspondentes

# Categorizar intensidade em 5 classes: muito baixa (0-51), baixa (52-102), média (103-153), alta (154-204), muito alta (205-255)
def categorize_intensity(value):
    if value < 0 or value > 255:
        raise ValueError("Value must be between 0 and 255.")
    
    step = 255 / 10  # Dividindo em 10 classes
    return int(value // step)  # Retorna a classe correspondente


# Processar imagens geradas
for img in average_images:  # 'average_images' é a lista de imagens do código anterior
    img_gray = img.convert("L")  # Converter para escala de cinza
    img_array = np.array(img_gray)  # Transformar em array NumPy
    
    for row in img_array:
        for pixel in row:
            pixel_data.append(pixel)
            labels.append(categorize_intensity(pixel))

# Converter listas para arrays NumPy
pixel_data = np.array(pixel_data).reshape(-1, 1)  # Pixels como características
labels = np.array(labels)                        # Rótulos como classes

# 2. Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    pixel_data, labels, test_size=0.3, random_state=42
)

# 3. Treinar o modelo SVM
print("Treinando o modelo SVM...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
print("Treinamento concluído.")

# 4. Avaliar o modelo
y_pred = svm_model.predict(X_test)
print("Avaliação do modelo:")
print(classification_report(y_test, y_pred))
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")

# 5. Fazer predições em uma nova imagem
print("Classificando pixels de uma nova imagem...")
new_image = average_images[0].convert("L")  # Exemplo: primeira imagem
new_image_array = np.array(new_image).reshape(-1, 1)  # Converter para formato adequado

# Classificar os pixels
predicted_classes = svm_model.predict(new_image_array)

# Reformatar para exibir como imagem
predicted_image = predicted_classes.reshape(new_image.size[::-1])

# Mostrar a imagem com as classes de intensidade
plt.figure(figsize=(8, 6))
plt.title("Classificação de Intensidade de Pixels")
plt.imshow(predicted_image, cmap='viridis')
plt.colorbar(label="Classes de Intensidade (0: Muito Baixa, 1: Baixa, 2: Média, 3: Alta, 4: Muito Alta)")
plt.show()

# Classificando pixels da última imagem
print("Classificando pixels da última imagem...")
last_image = average_images[-1].convert("L")  # Última imagem
last_image_array = np.array(last_image).reshape(-1, 1)  # Converter para formato adequado

# Classificar os pixels
predicted_classes_last = svm_model.predict(last_image_array)

# Reformatar para exibir como imagem
predicted_image_last = predicted_classes_last.reshape(last_image.size[::-1])

# Mostrar a última imagem com as classes de intensidade
plt.figure(figsize=(8, 6))
plt.title("Classificação de Intensidade de Pixels (Última Imagem)")
plt.imshow(predicted_image_last, cmap='viridis')
plt.colorbar(label="Classes de Intensidade (0: Muito Baixa, 1: Baixa, 2: Média, 3: Alta, 4: Muito Alta)")
plt.show()

# Mostrar ambas as imagens lado a lado
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Primeira imagem
axes[0].imshow(predicted_image, cmap='viridis')
axes[0].set_title("Classificação (Primeira Imagem)")
axes[0].axis('off')
axes[0].figure.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes[0], label="Classes de Intensidade")

# Última imagem
axes[1].imshow(predicted_image_last, cmap='viridis')
axes[1].set_title("Classificação (Última Imagem)")
axes[1].axis('off')
axes[1].figure.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes[1], label="Classes de Intensidade")

plt.tight_layout()
plt.show()

def draw_division_lines(image, center, title="Imagem com Divisões por Regiões"):
    """
    Desenha linhas divisórias na imagem com base em um ponto central.

    Args:
        image (np.ndarray): Imagem como array 2D.
        center (tuple): Coordenadas do ponto central (y, x).
        title (str): Título para a plotagem.

    Returns:
        None: Exibe a imagem com as divisões.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray", interpolation="nearest")
    
    # Ponto central
    center_y, center_x = center
    
    # Adiciona linhas divisórias
    plt.axhline(y=center_y, color="red", linestyle="--", linewidth=1)  # Linha horizontal
    plt.axvline(x=center_x, color="red", linestyle="--", linewidth=1)  # Linha vertical
    
    # Adiciona rótulos para as regiões
    plt.text(center_x + 5, center_y - 10, "NE", color="white", fontsize=12, bbox=dict(facecolor="red", alpha=0.5))
    plt.text(center_x + 5, center_y + 10, "SE", color="white", fontsize=12, bbox=dict(facecolor="red", alpha=0.5))
    plt.text(center_x - 30, center_y + 10, "SO", color="white", fontsize=12, bbox=dict(facecolor="red", alpha=0.5))
    plt.text(center_x - 30, center_y - 10, "NO", color="white", fontsize=12, bbox=dict(facecolor="red", alpha=0.5))
    plt.text(center_x - 40, center_y - 150, "N", color="white", fontsize=12, bbox=dict(facecolor="red", alpha=0.5))
    plt.text(center_x - 40, center_y + 150, "S", color="white", fontsize=12, bbox=dict(facecolor="red", alpha=0.5))
    plt.text(center_x + 150, center_y + 20, "L", color="white", fontsize=12, bbox=dict(facecolor="red", alpha=0.5))
    plt.text(center_x - 200, center_y + 20, "O", color="white", fontsize=12, bbox=dict(facecolor="red", alpha=0.5))
    
    plt.title(title)
    plt.axis("off")  # Ocultar os eixos
    plt.show()

def count_pixels_per_region(image, center):
    """
    Conta os pixels de cada classe em cada região.

    Args:
        image (np.ndarray): Imagem com classes como valores inteiros.
        center (tuple): Coordenadas do ponto central (y, x).

    Returns:
        dict: Contagem de pixels para cada classe em cada região.
    """
    height, width = image.shape
    center_y, center_x = center

    # Máscaras para as regiões
    regions = {
        'N': (slice(0, center_y), slice(center_x, width)),
        'S': (slice(center_y, height), slice(center_x, width)),
        'L': (slice(0, height), slice(center_x, width)),
        'O': (slice(0, height), slice(0, center_x)),
        'NE': (slice(0, center_y), slice(center_x, width)),
        'SE': (slice(center_y, height), slice(center_x, width)),
        'SO': (slice(center_y, height), slice(0, center_x)),
        'NO': (slice(0, center_y), slice(0, center_x)),
    }

    pixel_counts = {}
    for region, (rows, cols) in regions.items():
        region_pixels = image[rows, cols].flatten()
        unique, counts = np.unique(region_pixels, return_counts=True)
        pixel_counts[region] = dict(zip(unique, counts))

    return pixel_counts

# Escolha as imagens
first_image = predicted_image  # Primeira imagem classificada
last_image = predicted_image_last  # Última imagem classificada

# Defina o ponto central
center_point_first = (first_image.shape[0] // 2, first_image.shape[1] // 2)
center_point_last = (last_image.shape[0] // 2, last_image.shape[1] // 2)

# Desenhe as divisões nas imagens
draw_division_lines(first_image, center_point_first, title="Primeira Imagem com Divisões")
draw_division_lines(last_image, center_point_last, title="Última Imagem com Divisões")

# Conte os pixels de cada classe em cada região
pixel_counts_first = count_pixels_per_region(first_image, center_point_first)
pixel_counts_last = count_pixels_per_region(last_image, center_point_last)

# Exiba os resultados
print("Contagem de pixels por região (Primeira Imagem):")
for region, counts in pixel_counts_first.items():
    print(f"{region}: {counts}")

print("\nContagem de pixels por região (Última Imagem):")
for region, counts in pixel_counts_last.items():
    print(f"{region}: {counts}")

def draw_division_lines_side_by_side(image1, center1, image2, center2, title1="Imagem 1", title2="Imagem 2"):
    """
    Desenha linhas divisórias em duas imagens e exibe lado a lado.

    Args:
        image1 (np.ndarray): Primeira imagem como array 2D.
        center1 (tuple): Coordenadas do ponto central da primeira imagem (y, x).
        image2 (np.ndarray): Segunda imagem como array 2D.
        center2 (tuple): Coordenadas do ponto central da segunda imagem (y, x).
        title1 (str): Título da primeira imagem.
        title2 (str): Título da segunda imagem.

    Returns:
        None: Exibe as imagens lado a lado com as divisões.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Primeira imagem
    axes[0].imshow(image1, cmap="gray", interpolation="nearest")
    axes[0].axhline(y=center1[0], color="red", linestyle="--", linewidth=1)  # Linha horizontal
    axes[0].axvline(x=center1[1], color="red", linestyle="--", linewidth=1)  # Linha vertical
    axes[0].set_title(title1)
    axes[0].axis("off")

    # Segunda imagem
    axes[1].imshow(image2, cmap="gray", interpolation="nearest")
    axes[1].axhline(y=center2[0], color="red", linestyle="--", linewidth=1)  # Linha horizontal
    axes[1].axvline(x=center2[1], color="red", linestyle="--", linewidth=1)  # Linha vertical
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# Defina o ponto central para ambas as imagens
center_point_first = (first_image.shape[0] // 2, first_image.shape[1] // 2)
center_point_last = (last_image.shape[0] // 2, last_image.shape[1] // 2)

# Exiba as duas imagens com divisões lado a lado
draw_division_lines_side_by_side(
    first_image, center_point_first, 
    last_image, center_point_last, 
    title1="Primeira Imagem com Divisões", 
    title2="Última Imagem com Divisões"
)

def calculate_pixel_differences(counts_first, counts_last):
    """
    Calcula a diferença no número de pixels por classe entre duas imagens.

    Args:
        counts_first (dict): Contagem de pixels para cada classe na primeira imagem.
        counts_last (dict): Contagem de pixels para cada classe na última imagem.

    Returns:
        pd.DataFrame: DataFrame com as diferenças por região e classe.
    """
    regions = counts_first.keys()
    classes = sorted(set(k for r in counts_first.values() for k in r.keys()))
    
    differences = []
    for region in regions:
        diff = {
            "Region": region,
            **{f"Class_{cls}": counts_last[region].get(cls, 0) - counts_first[region].get(cls, 0) for cls in classes}
        }
        differences.append(diff)

    return pd.DataFrame(differences)

def plot_pixel_differences(differences_df):
    """
    Plota um gráfico de barras empilhadas para as diferenças de pixels por região e classe.

    Args:
        differences_df (pd.DataFrame): DataFrame com as diferenças por região e classe.
    """
    regions = differences_df["Region"]
    class_columns = [col for col in differences_df.columns if col.startswith("Class_")]

    # Transforma os dados para um gráfico de barras empilhadas
    differences_df.set_index("Region")[class_columns].plot(
        kind="bar", stacked=True, figsize=(12, 6), colormap="viridis"
    )
    plt.title("Diferença no Número de Pixels por Classe e Região")
    plt.xlabel("Regiões")
    plt.ylabel("Diferença de Pixels")
    plt.legend(title="Classes")
    plt.tight_layout()
    plt.show()

# Calcular as diferenças de pixels
differences_df = calculate_pixel_differences(pixel_counts_first, pixel_counts_last)

# Exibir o DataFrame para ver as diferenças (opcional)
print(differences_df)

# Plotar o gráfico de barras empilhadas
plot_pixel_differences(differences_df)
