import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from image_processing import process_image

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        original_image = Image.open(file_path)
        processed_image = process_image(original_image)
        display_image(processed_image)

def display_image(image):
    img_label.configure(image=image)
    img_label.image = image

# Create the main application window
root = tk.Tk()
root.title("Image Processing App")

# Button to open image
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

# Label to display image
img_label = tk.Label(root)
img_label.pack()

root.mainloop()