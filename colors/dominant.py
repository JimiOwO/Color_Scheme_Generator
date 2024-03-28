# %%
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys


# %%
quantization_factor = 5

# %%
def get_dominant_color(image: np.array):
  # Ensure the image is 3D array
  if len(image.shape) != 3:
    raise ValueError("The image should be a 3D array")

  # Use L*a*b color space to get the dominant color
  cie_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

  # Quantize the colors
  cie_lab = cie_lab // quantization_factor * quantization_factor

  # Get the most common color
  colors, counts = np.unique(cie_lab.reshape(-1, cie_lab.shape[2]), axis=0, return_counts=True)
  most_common_color = colors[np.argmax(counts)]

  return most_common_color

# %%
def slice_image(image: np.array, block_size: int = 10):
  # Ensure the image is 3D array
  if len(image.shape) != 3:
    raise ValueError("The image should be a 3D array")

  result = []

  # Slice the image into multiple block_size * block_size pixel blocks
  for i in range(0, image.shape[0], block_size):
    for j in range(0, image.shape[1], block_size):
      # Check if indices are within image size
      end_row = i + block_size
      end_col = j + block_size
      if i + block_size > image.shape[0]:
        end_row = image.shape[0]
      if j + block_size > image.shape[1]:
        end_col = image.shape[1]

      # Get the block
      block = image[i:end_row, j:end_col]
      result.append(get_dominant_color(block))

  return result

# %%
def load_image(image_path):
    image = cv2.imread(image_path)
    return image


# Create Image
def create_image(image_name):
  output_name = image_name.split("/")[-1].split(".")[0] + ".png"
  image = load_image(image_name)
  labs = slice_image(image, 10)

  lab_array = np.array(labs)

  # Get the most common color
  colors, counts = np.unique(lab_array, axis=0, return_counts=True)

  # 10 most common colors with their counts
  most_common_colors = colors[np.argsort(counts)[::-1]][:10]
  most_common_colors_counts = counts[np.argsort(counts)[::-1]][:10]

  # Change the most common colors to RGB
  most_common_colors_rgb = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_LAB2RGB)[0][0] for color in most_common_colors]

  # Plot histogram of the most common colors

  plt.figure(figsize=(15, 5))
  plt.bar(range(10), most_common_colors_counts, color=[(c[0] / 255, c[1] / 255, c[2] / 255) for c in most_common_colors_rgb])
  plt.xticks(range(10), [f"{most_common_colors_rgb[i]}" for i in range(10)])
  plt.xlabel("Color")
  plt.ylabel("Count")
  plt.title("Most common colors")
  plt.savefig(f"./output/{output_name}")
  # plt.show()
  

if __name__ == "__main__":
  # Create output directory if not exists
  if not os.path.exists("./output"):
    os.makedirs("./output")

  # Extract argument
  images = sys.argv[1:]
  for image in images:
    create_image(image)





