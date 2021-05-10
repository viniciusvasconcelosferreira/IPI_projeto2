import numpy as np
import cv2 as cv
# Bibliotecas externas usadas para
# Image IO
from PIL import Image

# Filtragem morfológica
from skimage.morphology import opening
from skimage.morphology import disk

def improve_image_quality():
    black = 0
    white = 255
    threshold = 160

    # Abertura da imagem de entrada no modo de tons de cinza e obtenha seus pixels.
    img = Image.open("imagens/entrada/morf_test.png").convert("LA")
    pixels = np.array(img)[:, :, 0]

    # Remoção dos pixels acima do limite
    pixels[pixels > threshold] = white
    pixels[pixels < threshold] = black

    # Abertura morfológica
    blobSize = 1  # Seleção do raio máximo dos blobs para remover
    structureElement = disk(blobSize)  # definição da forma
    # Inverção da imagem de forma que o preto seja o fundo e o primeiro plano branco para realizar a abertura
    pixels = np.invert(opening(np.invert(pixels), structureElement))

    # Criação e salvamento de uma nova imagem
    newImg = Image.fromarray(pixels).convert('RGB')
    newImg.save("imagens/saida/morf_test_improve_image_quality.png")

    # Encontro dos componentes conectados (objetos pretos na imagem)
    # Como a função procura componentes brancos conectados em um fundo preto, precisamos inverter a imagem
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(np.invert(pixels), connectivity=8)

    # Para cada componente conectado na imagem, será obtido o número de pixels da variável de estatísticas no último
    # coluna. Assim, removemos a primeira entrada de tamanhos, porque esta é a entrada do componente conectado em segundo plano
    sizes = stats[1:, -1]
    nb_components -= 1

    # Definição do tamanho mínimo (número de pixels) em que um componente deve consistir
    minimum_size = 100

    # Criação de uma nova imageml
    newPixels = np.ones(pixels.shape) * 255

    # Iteração sobre todos os componentes da imagem para manter os componentes maiores do que o tamanho mínimo
    for i in range(1, nb_components):
        if sizes[i] > minimum_size:
            newPixels[output == i + 1] = 0

    # Criação e salvamento de uma nova imagem
    newImg = Image.fromarray(newPixels).convert('RGB')
    newImg.save("imagens/saida/morf_test_improve_image_quality(1).png")


def improve_image_quality_compare_every_pixel():
    black = (0, 0, 0)
    white = (255, 255, 255)
    threshold = (160, 160, 160)

    # Abertura da imagem na escala de cinza e obtenção dos pixels
    img = Image.open("imagens/entrada/morf_test.png").convert("LA")
    pixels = img.getdata()

    newPixels = []

    # Comparação de cada pixel
    for pixel in pixels:
        if pixel < threshold:
            newPixels.append(black)
        else:
            newPixels.append(white)

    # Criação e salvamento de uma nova imagem
    newImg = Image.new("RGB", img.size)
    newImg.putdata(newPixels)
    newImg.save("imagens/saida/morf_test_improve_image_quality.png")
