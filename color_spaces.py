import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def RGB_color_space(imagem):
    r, g, b = cv.split(imagem)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = imagem.reshape((np.shape(imagem)[0] * np.shape(imagem)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Vermelho")
    axis.set_ylabel("Verde")
    axis.set_zlabel("Azul")
    plt.savefig('imagens/saida/image_in_rgb_color_space.png')
    return True


def HSV_color_space(imagem):
    nova_imagem = cv.cvtColor(imagem, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(nova_imagem)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = imagem.reshape((np.shape(imagem)[0] * np.shape(imagem)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Matiz")
    axis.set_ylabel("Saturação")
    axis.set_zlabel("Brilho")
    plt.savefig('imagens/saida/image_in_hsv_color_space.png')
    return True
