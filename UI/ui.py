import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os,sys
import cv2
sys.path.append(os.getcwd())
from colors.standout import get_n_standing_out_color,create_overlay

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