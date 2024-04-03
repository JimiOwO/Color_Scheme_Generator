# %%
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
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


def get_standing_out_color(colors: np.array, n: int = 10):
    points = [color for color in colors]
    distances_list = []
    for i in range(len(points)):
        for j in range(len(points)):
            distances_list.append(
                (calculate_distance(points[i], points[j]), i, j))

    distances_list = sorted(distances_list, key=lambda x: x[0])
    uf = UnionFind(len(points))
    ncomp = len(points)
    for (_, i, j) in distances_list:
        if not uf.is_same_set(i, j):
            uf.union(i, j)
            ncomp -= 1
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


def create_dominant_image_graph(image_name):
    # Split with either / or \
    output_name = image_name.split(
        "/")[-1].split("\\")[-1].split(".")[0] + ".png"
    print("output = ", output_name)
    image = load_image(image_name)
    labs = slice_image(image, 10)

    lab_array = np.array(labs)

    # Max l , a, b
    # max_l = lab_array[:, 0].max()
    # max_a = lab_array[:, 1].max()
    # max_b = lab_array[:, 2].max()
    # min_l = lab_array[:, 0].min()
    # min_a = lab_array[:, 1].min()
    # min_b = lab_array[:, 2].min()
    # print(max_l, max_a, max_b)
    # print(min_l, min_a, min_b)

    # Get the most common color
    colors, counts = np.unique(lab_array, axis=0, return_counts=True)

    # max_l = colors[:, 0].max()
    # max_a = colors[:, 1].max()
    # max_b = colors[:, 2].max()
    # min_l = colors[:, 0].min()
    # min_a = colors[:, 1].min()
    # min_b = colors[:, 2].min()
    # print(max_l, max_a, max_b)
    # print(min_l, min_a, min_b)
    # Save to csv
    # np.savetxt(f"./output/{output_name}.csv", colors, delimiter=",")

    print(len(colors))
    standingout_colors = get_standing_out_color(colors, 10)
    standingout_colors_rgb = [cv2.cvtColor(
        np.uint8([[color]]), cv2.COLOR_LAB2RGB)[0][0] for color in standingout_colors]

    print(standingout_colors)
    print(standingout_colors_rgb)

    plt.figure(figsize=(15, 5))
    plt.bar(range(10), [1 for _ in standingout_colors], color=[
            (c[0] / 255, c[1] / 255, c[2] / 255) for c in standingout_colors_rgb])
    plt.xticks(range(10), [f"{standingout_colors_rgb[i]}" for i in range(10)])
    plt.xlabel("Color")
    plt.ylabel("Count")
    plt.title("Most common colors")
    plt.savefig(f"./output/standing_{output_name}")
    # plt.show()


if __name__ == "__main__":
    # Create output directory if not exists
    if not os.path.exists("./output"):
        os.makedirs("./output")

    # Extract argument
    images = sys.argv[1:]
    for image in images:
        create_dominant_image_graph(image)
