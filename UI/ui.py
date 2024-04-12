import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os,sys
sys.path.append(os.getcwd())
from colors.standout import get_n_standing_out_color,create_dominant_image_graph

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        width, height = image.size
        max_width = 720
        max_height = 480
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height))

        photo_image = ImageTk.PhotoImage(image)
        img_label.configure(image=photo_image)
        img_label.image = photo_image
        color_list = get_n_standing_out_color(file_path)
        create_dominant_image_graph(color_list)
            
root = tk.Tk()
root.title("color scheme generator")
root.geometry("720x480")
txt = tk.Label(root,text="please open the image to generate color scheme.",font=("Times New Roman",24))
open_button = tk.Button(root, text="Open Image", command=open_image,font=("Times New Roman",24))
txt.pack()
open_button.pack()
img_label = tk.Label(root)
img_label.pack()
root.mainloop()