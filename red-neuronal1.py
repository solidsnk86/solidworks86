{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyMwIxdqL8fdtqaZkYKNrjer",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/solidsnk86/solidworks86/blob/main/red-neuronal1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "TGU8xlPADgXn"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "import numpy as np"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "celsius = np.array([ -40, -10, 0, 8, 15, 22, 38], dtype=float)\n",
        "fahrenheit = np.array([ -40, -14, 32, 46, 59, 72, 100], dtype=float)"
      ],
      "metadata": {
        "id": "VsTjDxzLFO7b"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# capa = tf.keras.layers.Dense(units=1, input_shape=[1])\n",
        "# modelo = tf.keras.Sequential([capa])\n",
        "\n",
        "oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])\n",
        "oculta2 = tf.keras.layers.Dense(units=3)\n",
        "salida = tf.keras.layers.Dense(units=1)\n",
        "modelo = tf.keras.Sequential([oculta1, oculta2, salida])"
      ],
      "metadata": {
        "id": "ZsQRL7zoLhNB"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "modelo.compile(\n",
        "    optimizer=tf.keras.optimizers.Adam(0.1),\n",
        "    loss='mean_squared_error')"
      ],
      "metadata": {
        "id": "P2lKT0q8Evl9"
      },
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Comenzando Entrenamiento...\")\n",
        "historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)\n",
        "print(\"Modelo Entrenado\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V_SJazRsFFLv",
        "outputId": "3650ce68-7aa2-4e2f-afb1-d2d7e5c0f825"
      },
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Comenzando Entrenamiento...\n",
            "Modelo Entrenado\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "plt.xlabel(\"# Epoca\")\n",
        "plt.ylabel(\"Magnitud de Pérdida\")\n",
        "plt.plot(historial.history[\"loss\"])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "F0vPT_owFebb",
        "outputId": "5369e428-a8b5-4196-bcfe-5cc824f43c94"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7fd0ad7e7370>]"
            ]
          },
          "metadata": {},
          "execution_count": 32
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Hagamos una Predicción\")\n",
        "resultado = modelo.predict([100.0])\n",
        "print(\"El resultado es \" + str(resultado) + \"fahrenheit!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "p_cCcw9TGjaT",
        "outputId": "26683c78-0c9d-450b-894e-99d00825d25e"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Hagamos una Predicción\n",
            "1/1 [==============================] - 0s 65ms/step\n",
            "El resultado es [[218.18433]]fahrenheit!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Variables internas del modelo\")\n",
        "#print(capa.get_weights())\n",
        "\n",
        "print(oculta1.get_weights())\n",
        "print(oculta2.get_weights())\n",
        "print(salida.get_weights())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "w8ndOaQuGjT8",
        "outputId": "1cf55cd0-d8fc-4ac7-90b0-7a8ac54bd99d"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Variables internas del modelo\n",
            "[array([[ 0.46484697, -0.06749261,  0.5828987 ]], dtype=float32), array([3.581993 , 0.5705583, 2.868476 ], dtype=float32)]\n",
            "[array([[-0.6273746 ,  0.80785465,  1.844768  ],\n",
            "       [-0.3584633 , -0.8321593 ,  0.13230872],\n",
            "       [ 0.400829  , -0.8516386 ,  0.41423038]], dtype=float32), array([-2.3425946,  1.8716826,  4.9837055], dtype=float32)]\n",
            "[array([[-0.12812617],\n",
            "       [ 0.21683745],\n",
            "       [ 1.758791  ]], dtype=float32), array([3.9586556], dtype=float32)]\n"
          ]
        }
      ]
    }
  ]
}