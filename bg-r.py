import cv2

import numpy as np

from tkinter import *

def remove_background(image):


  grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


  threshold_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]


  contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


  largest_contour = max(contours, key=cv2.contourArea)

  

  mask = np.zeros(image.shape, dtype=np.uint8)

  cv2.drawContours(mask, [largest_contour], 0, 255, -1)


  masked_image = cv2.bitwise_and(image, image, mask=mask)

  return masked_image



root = Tk()



label = Label(root)



button_load = Button(root, text="Load Image", command=load_image)



button_download = Button(root, text="Download Image", command=download_image)



label.pack()

button_load.pack()

button_download.pack()

# Comienzo del loop

root.mainloop()

def load_image():

# Creamos ventana de carga
  
  path = askopenfilename()

  # Carga de la imagen y lectura con Open CV

  image = cv2.imread(path)

  # Mostrar la imagen en la etiqueta 

  label.config(image=image)

def download_image():

  # Obtener etiquetas de imagen 

  image = label.cget("image")

  # Guardar imágenes como archivo jpg

  cv2.imwrite("image_without_background.jpg", image)

  
