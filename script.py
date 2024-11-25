from sklearn.model_selection import train_test_split
from functions import calculate_yearly_average_images, classify_image_pixels_with_svm, count_pixels_by_categorized_intensity, create_svm_model, divide_image, convert_yearly_average_images_dict_to_list, get_initial_file_variables, get_raw_image_file_paths, get_images_grouped_by_year, plot_region_comparisons, prepare_svm_data, show_first_and_last_image_classification_with_svm, show_image_parts, sort_image_file_paths, show_all_images, test_svm_model, train_svm_model

# Control variables
image_band = 3
enable_bnu = True

# Initial file handling
file_name_pattern, files_path = get_initial_file_variables(enable_bnu)

file_paths = get_raw_image_file_paths(file_name_pattern, files_path, ["7_2022_BNU_250.tif", "07_2022_UTM.tif"])
file_paths = sort_image_file_paths(file_paths, file_name_pattern, files_path)

show_all_images(file_name_pattern, file_paths, image_band)

images_grouped_by_year = get_images_grouped_by_year(file_name_pattern, file_paths, image_band)
yearly_average_images_dict = calculate_yearly_average_images(images_grouped_by_year, False)

# -------------------------
# SVM
# -------------------------

# 2014
#   monthly_images
#   year_average_image
#   year_average_image_pil

yearly_average_images, yearly_average_images_pil = convert_yearly_average_images_dict_to_list(yearly_average_images_dict)

pixel_data, labels = prepare_svm_data(yearly_average_images_pil)

X_train, X_test, y_train, y_test = train_test_split(
    pixel_data, labels, test_size=0.3, random_state=42
)

svm_model = create_svm_model()
train_svm_model(svm_model, X_train, y_train)

y_pred = test_svm_model(svm_model, X_test, y_test, False)

first_image = yearly_average_images[0]
last_image = yearly_average_images[-1]

first_image_pil = yearly_average_images_pil[0]
last_image_pil = yearly_average_images_pil[-1]

first_image_classified_with_svm = classify_image_pixels_with_svm(svm_model, first_image_pil)
last_image_classified_with_svm = classify_image_pixels_with_svm(svm_model, last_image_pil)

show_first_and_last_image_classification_with_svm(first_image_classified_with_svm, last_image_classified_with_svm)

# Separar imagem
first_top, first_middle, first_bottom = divide_image(first_image_classified_with_svm)
last_top, last_middle, last_bottom = divide_image(last_image_classified_with_svm)

show_image_parts(first_top, first_middle, first_bottom)
show_image_parts(last_top, last_middle, last_bottom)

# Contar os pixels por categoria de intensidade para cada parte
first_top_pixels_count = count_pixels_by_categorized_intensity(first_top)
first_middle_pixels_count = count_pixels_by_categorized_intensity(first_middle)
first_bottom_pixels_count = count_pixels_by_categorized_intensity(first_bottom)

last_top_pixels_count = count_pixels_by_categorized_intensity(last_top)
last_middle_pixels_count = count_pixels_by_categorized_intensity(last_middle)
last_bottom_pixels_count = count_pixels_by_categorized_intensity(last_bottom)

# Exibir uma comparação entre as imagens

# Comparar os pixels da região superior
plot_region_comparisons(first_top_pixels_count, last_top_pixels_count, "Top")

# Comparar os pixels da região intermediária
plot_region_comparisons(first_middle_pixels_count, last_middle_pixels_count, "Middle")

# Comparar os pixels da região inferior
plot_region_comparisons(first_bottom_pixels_count, last_bottom_pixels_count, "Bottom")



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Definições das classes e dados agrupados
x = np.arange(10)  # Classes de intensidade (0 a 9)
data_pairs = [
    (first_top_pixels_count, last_top_pixels_count),
    (first_middle_pixels_count, last_middle_pixels_count),
    (first_bottom_pixels_count, last_bottom_pixels_count),
]
titles = ["Top Slices", "Middle Slices", "Bottom Slices"]

# Configuração do layout para 3 gráficos
fig, axes = plt.subplots(1, 3, figsize=(10, 18))
bars_list = []
for ax, title in zip(axes, titles):
    bars = ax.bar(x, [0] * 10, color='skyblue')
    ax.set_title(title)
    ax.set_xlabel("Intensity classes (0-9)")
    ax.set_ylabel("Amount of pixels")
    ax.set_xticks(x)
    ax.set_ylim(0, max([max(d.values()) for d in [first_top_pixels_count, last_top_pixels_count]]) + 500)
    bars_list.append(bars)

# Função para atualizar os gráficos
def update(frame):
    for i, (bars, (data_first, data_last)) in enumerate(zip(bars_list, data_pairs)):
        data = data_first if frame % 2 == 0 else data_last
        for bar, height in zip(bars, data.values()):
            bar.set_height(height)
        axes[i].set_title(f"{titles[i]} - {'First' if frame % 2 == 0 else 'Last'}")

# Animação
ani = FuncAnimation(fig, update, frames=6, interval=1000, repeat=True)
plt.tight_layout()
plt.show()
