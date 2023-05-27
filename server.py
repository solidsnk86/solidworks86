from flask import Flask, request, jsonify

import cv2

import numpy as np

from mrcnn import utils

from mrcnn import model as modellib

app = Flask(__name__)

# Configuración del modelo

class InferenceConfig(Config):

    NAME = "inference_config"

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 2  # Fondo y objeto de interés

    # Resto de parámetros de configuración

# Directorio de los pesos pre-entrenados del modelo Mask R-CNN

WEIGHTS_PATH = "path/to/pretrained/weights.h5"

# Cargar la configuración y los pesos del modelo

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")

model.load_weights(WEIGHTS_PATH, by_name=True)

# Función para eliminar el fondo de una imagen

def remove_background(image_path):

    image = cv2.imread(image_path)

    results = model.detect([image], verbose=0)

    r = results[0]

    # Crear una máscara binaria para el fondo

    background_mask = np.where(r['masks'][:,:,0], 0, 255).astype(np.uint8)

    # Aplicar la máscara al fondo de la imagen

    image = cv2.bitwise_and(image, image, mask=background_mask)

    return image

# Ruta para manejar la solicitud POST de eliminación de fondo

@app.route('/eliminar_fondo', methods=['POST'])

def eliminar_fondo():

    # Verificar si se envió una imagen

    if 'image' not in request.files:

        return jsonify({'result': 'error', 'message': 'No se envió ninguna imagen'})

    # Recibir la imagen enviada por el cliente

    image = request.files['image']

    # Guardar la imagen en disco

    image_path = 'temp_image.jpg'

    image.save(image_path)

    try:

        # Procesar la imagen y eliminar el fondo utilizando tu código de eliminación de fondo

        processed_image = remove_background(image_path)

        # Eliminar la imagen temporal

        os.remove(image_path)

        # Devolver la imagen procesada como una respuesta JSON

        _, processed_image_data = cv2.imencode('.jpg', processed_image)

        processed_image_base64 = base64.b64encode(processed_image_data).decode('utf-8')

        return jsonify({'result': 'success', 'image': processed_image_base64})

    except Exception as e:

        return jsonify({'result': 'error', 'message': str(e)})

if __name__ == '__main__':

    app.run()

    
