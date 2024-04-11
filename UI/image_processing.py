from PIL import Image
from PIL import ImageTk

def process_image(image):
    # Example processing operation: Convert image to grayscale
    processed_image = image.convert("L")
    return ImageTk.PhotoImage(processed_image)
