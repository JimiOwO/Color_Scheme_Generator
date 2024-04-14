import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os,sys
import cv2
sys.path.append(os.getcwd())
from colors.standout import get_n_standing_out_color,create_dominant_image_graph

def process_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        width, height = image.size
        max_width = 1366
        max_height = 768
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height))
        color_list = get_n_standing_out_color(file_path)
        result = create_overlay(color_list, image)
        photo_image = ImageTk.PhotoImage(result)
        img_label.configure(image=photo_image)
        img_label.image = photo_image


# Function to create overlay with cubes and white outline
def create_overlay(cube_colors, image):
    # Convert PIL image to NumPy array
    image_array = np.array(image)
    separation_factor = 1/3
    cube_size = int(0.075 * image_array.shape[1])
    separation = int(cube_size * separation_factor)
    margin = 30
    outline_thickness = 1
    
    x_start = int(margin*1.5)
    y_start = int(image_array.shape[0] - cube_size - outline_thickness*2 - margin*1.5)

    # Loop through each cube color
    for color in cube_colors:
        # Create cube with white outline
        cube_with_outline = np.full((cube_size + 2 * outline_thickness, cube_size + 2 * outline_thickness, 3), 255, dtype=np.uint8)
        cube_with_outline[outline_thickness:-outline_thickness, outline_thickness:-outline_thickness] = color

        # Overlay the colored cube with white outline on the image
        image_array[y_start:y_start+cube_size + 2 * outline_thickness, 
                    x_start:x_start+cube_size + 2 * outline_thickness] = cube_with_outline

        # Update x_start for the next cube
        x_start += cube_size + separation

    # Trim the image to its original size
    image_array = image_array[margin:-margin, margin:-margin]

    return Image.fromarray(image_array)
            
root = tk.Tk()
root.title("color scheme generator")
# root.geometry("1366x768")
txt = tk.Label(root,text="Select image to generate palette.",font=("Arial",24))
open_button = tk.Button(root, text="Open Image", command=process_image,font=("Arial",24))
txt.pack()
open_button.pack()
img_label = tk.Label(root)
img_label.pack()
root.mainloop()