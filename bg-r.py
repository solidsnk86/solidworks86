import cv2

import numpy as np

from tkinter import *

def remove_background(image):

  # Convert the image to grayscale

  grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Threshold the image to create a binary image

  threshold_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]

  # Find the contours in the binary image

  contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Find the largest contour

  largest_contour = max(contours, key=cv2.contourArea)

  # Create a mask of the largest contour

  mask = np.zeros(image.shape, dtype=np.uint8)

  cv2.drawContours(mask, [largest_contour], 0, 255, -1)

  # Mask the image with the contour

  masked_image = cv2.bitwise_and(image, image, mask=mask)

  return masked_image

# Create a Tkinter window

root = Tk()

# Create a label to display the image

label = Label(root)

# Create a button to load an image

button_load = Button(root, text="Load Image", command=load_image)

# Create a button to download the image without the background removed

button_download = Button(root, text="Download Image", command=download_image)

# Place the label and buttons on the window

label.pack()

button_load.pack()

button_download.pack()

# Start the main loop

root.mainloop()

def load_image():

  # Get the path to the image

  path = askopenfilename()

  # Load the image

  image = cv2.imread(path)

  # Display the image on the label

  label.config(image=image)

def download_image():

  # Get the image from the label

  image = label.cget("image")

  # Save the image to a file

  cv2.imwrite("image_without_background.jpg", image)

  
