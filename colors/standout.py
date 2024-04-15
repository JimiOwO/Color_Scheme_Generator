# %%
from math import sqrt
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import sys
from PIL import Image


# %%
quantization_factor = 5
good_range = 6

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
    colors, counts = np.unique(
        cie_lab.reshape(-1, cie_lab.shape[2]), axis=0, return_counts=True)
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


def convert_LAB_to_point3d(lab):
    return np.array([lab[0], lab[1], lab[2]])


def calculate_distance(point1, point2):
    return np.linalg.norm(
        convert_LAB_to_point3d(point1) - convert_LAB_to_point3d(point2))


class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0 for _ in range(n)]
        self.mapping = {}
        self.last_id = 0

    def register(self, x):
        if x not in self.mapping:
            self.mapping[x] = self.last_id
            self.last_id += 1
        return self.mapping[x]

    def find(self, x):
        x = self.register(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def is_same_set(self, x, y):
        x = self.register(x)
        y = self.register(y)
        return self.find(x) == self.find(y)

    def union(self, x, y):
        x = self.register(x)
        y = self.register(y)
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return

        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1

    def get_groups(self):
        groups = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups


def reduce_colors(colors: np.array, n: int = -1):
    points = [color for color in colors]
    distances_list = []
    for i in range(len(points)):
        for j in range(len(points)):
            distances_list.append(
                (calculate_distance(points[i], points[j]), i, j))

    distances_list = sorted(distances_list, key=lambda x: x[0])
    uf = UnionFind(len(points))
    ncomp = len(points)

    first_dist = - 1

    for (dist, i, j) in distances_list:
        if not uf.is_same_set(i, j):
            uf.union(i, j)
            ncomp -= 1
        if n == -1:
            if first_dist == -1 and dist != 0:
                first_dist = dist
            if dist != 0 and dist > first_dist:
                first_dist = dist
                if ncomp <= good_range:
                    break
        if ncomp == n:
            break

    groups = uf.get_groups()
    standing_out_colors = []
    for group in groups.values():
        group_colors = [colors[i] for i in group]
        # USING KMedoids
        kmedoids = KMedoids(n_clusters=1).fit(group_colors)
        standing_out_colors.append(kmedoids.cluster_centers_[0])

    return standing_out_colors


# Create Image
def get_n_standing_out_color(image_name, n: int = -1):
    image = load_image(image_name)

    min_dim = min(image.shape[0], image.shape[1]) // 100

    labs = slice_image(image, min_dim)

    lab_array = np.array(labs)

    # Get the most common color
    colors, counts = np.unique(lab_array, axis=0, return_counts=True)

    while (len(colors) > max(100, n * n)):
        size = len(colors)
        colors = reduce_colors(colors, int(sqrt(size)))

    if n != -1:
        colors = reduce_colors(colors, n)
    else:
        colors = reduce_colors(colors)

    standingout_colors = colors
    standingout_colors_rgb = [cv2.cvtColor(
        np.uint8([[color]]), cv2.COLOR_LAB2RGB)[0][0] for color in standingout_colors]

    return standingout_colors_rgb


def create_dominant_image_graph(standingout_colors, output_name="output.png"):

    len_colors = len(standingout_colors)
    plt.figure(figsize=(15, 5))
    plt.bar(range(len_colors), [1 for _ in standingout_colors], color=[
            (c[0] / 255, c[1] / 255, c[2] / 255) for c in standingout_colors])
    plt.xticks(range(len_colors), [
               f"{standingout_colors[i]}" for i in range(len_colors)])
    plt.xlabel("Color")
    plt.ylabel("Count")
    plt.title("Most common colors")
    plt.savefig(f"./output/standing_{output_name}")
    plt.show()


def create_overlay(colors, image):
    # Convert PIL image to NumPy array
    image_array = np.array(image)
    separation_factor = 1/3
    cube_size = int(0.075 * image_array.shape[1])
    separation = int(cube_size * separation_factor)
    margin = 30

    x_start = int(margin*1.5)
    y_start = int(image_array.shape[0] - cube_size - 2 - margin*1.5)

    # Loop through each cube color
    for color in colors:
        # Create cube with white outline
        cube_with_outline = np.full(
            (cube_size + 2 * 1, cube_size + 2 * 1, 3), 255, dtype=np.uint8)
        cube_with_outline[1:-1, 1:-1] = color

        # Overlay the colored cube with white outline on the image
        image_array[y_start:y_start+cube_size + 2 * 1,
                    x_start:x_start+cube_size + 2 * 1] = cube_with_outline

        # Update x_start for the next cube
        x_start += cube_size + separation

    # Trim the image to its original size
    image_array = image_array[margin:-margin, margin:-margin]

    return Image.fromarray(image_array)


if __name__ == "__main__":
    # Create output directory if not exists
    if not os.path.exists("./output"):
        os.makedirs("./output")

    # Extract argument
    images = sys.argv[1:]
    for image in images:
        output_name = image.split(
            "/")[-1].split("\\")[-1].split(".")[0] + ".png"
        standingout_colors = get_n_standing_out_color(image)
        create_dominant_image_graph(
            standingout_colors, output_name=output_name)
