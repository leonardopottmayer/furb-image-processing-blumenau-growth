import matplotlib.pyplot as plt

# Show an image with a custom title (matplotlib compatibility).
def show_image(title, img):
    # plt.imshow(img, cmap='gray')
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()