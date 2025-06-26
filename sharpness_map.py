import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def compute_laplacian_sharpness(gray):
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    abs_lap = np.abs(laplacian)
    sharpness_map = cv2.normalize(abs_lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    score = laplacian.var()
    return sharpness_map, score

def main():
    image_path = input("Enter image path (JPG/PNG): ").strip()

    if not os.path.exists(image_path):
        print("❌ File not found.")
        return

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lap_map, lap_score = compute_laplacian_sharpness(gray)
    print(f"Laplacian Sharpness Score: {lap_score:.2f}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image_rgb)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(lap_map, cmap='jet')
    axs[1].set_title(f"Laplacian Heatmap (Score: {lap_score:.2f})")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

    Image.fromarray(image_rgb).save("original_image.png")
    Image.fromarray(lap_map).save("laplacian_heatmap.png")

    print("Saved:")
    print(" - original_image.png")
    print(" - laplacian_heatmap.png")

if __name__ == "__main__":
    main()
