"""
Departamento de Ciência da Computação
Universidade de Brasília - UnB
Aluno: Vinicius Vasconcelos Ferreira
Matrícula: 18/0043358
Disciplina: Introdução ao Processamento de Imagens - IPI
Prof.º: Bruno Macchiavello
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import color_spaces as cp
import skin_detection as sd

"""
Primeiro passo (Detecção de pele)
 -  Normalização do modelo RGB

"""


def quest1_2():
    imagem = cv.imread('imagens/entrada/morf_test.png')
    median = cv.medianBlur(imagem, 5)
    cv.imwrite('imagens/saida/morf_test_skin_detection.png', median)


def quest1():
    # leitura da imagem
    imagem = cv.imread('imagens/entrada/morf_test.png')
    # verificar se a imagem é colorida e passar para escala de cinza
    if len(imagem.shape) > 2:
        imagem = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    # aplicação blur para suavização da imagem com 7x7
    suave = cv.GaussianBlur(imagem, (7, 7), 0)
    # Limiar adaptativo usando 21 pixels vizinhos mais próximos()
    bin1 = cv.adaptiveThreshold(suave, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 5)
    # impressão da imagem
    cv.imwrite('imagens/saida/morf_test_binarizada.png', bin1)


def quest2():
    # leitura da imagem
    imagem = cv.imread('imagens/entrada/cookies.tif')
    # verificar se a imagem é colorida e passar para escala de cinza
    if len(imagem.shape) > 2:
        imagem = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    # aplicação blur para suavização da imagem com 7x7
    suave = cv.GaussianBlur(imagem, (7, 7), 0)
    # Limiar adaptativo usando 21 pixels vizinhos mais próximos
    bin1 = cv.adaptiveThreshold(suave, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 5)
    # Limiar adaptativo usando 21 pixels vizinhos mais próximos (INVERTIDA)
    bin2 = cv.adaptiveThreshold(suave, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 5)
    # impressão da imagem
    cv.imwrite('imagens/saida/cookies_binarizada.tif', bin1)
    cv.imwrite('imagens/saida/cookies_binarizada_invertida.tif', bin2)
    # definição do kernel para uso dos algoritmos morfologicos
    kernel = np.ones((7, 7), np.uint8)
    # aplicação do algoritmo de erode para melhor visualização e uso de transformações
    erode_image = cv.erode(bin2, kernel, iterations=1)
    # impressão da imagem
    cv.imwrite('imagens/saida/cookies_binarizada_erode_transformation.tif', erode_image)
    # criação de máscara para eliminar o 'cookie' mordido
    mascara = np.zeros(erode_image.shape[:2], dtype='uint8')
    height, width = mascara.shape
    upper_left = (width // 2, height // 1000)
    bottom_right = (width * 3 // 1000, height * 3 // 2)
    cv.rectangle(mascara, upper_left, bottom_right, (255, 255, 255), thickness=-1)
    cv.imwrite('imagens/saida/cookies_binarizada_erode_mascara.tif', mascara)
    # aplicação da máscara sobre a imagem
    erode_image_mascara = cv.bitwise_and(erode_image, erode_image, mask=mascara)
    cv.imwrite('imagens/saida/cookies_binarizada_erode_mascara_aplicada.tif', erode_image_mascara)
    # remoção da máscara sobre a imagem atráves da inversão da máscara primária
    mascara_inversa = np.zeros(erode_image.shape[:2], dtype='uint8')
    height, width = mascara_inversa.shape
    upper_left = (width // 2, height // 1000)
    bottom_right = (width * 3 // 1000, height * 3 // 2)
    cv.rectangle(mascara_inversa, upper_left, bottom_right, (0, 0, 0), thickness=-1)
    erode_image_sem_mascara = cv.bitwise_not(erode_image, erode_image, mask=mascara_inversa)
    cv.imwrite('imagens/saida/cookies_binarizada_erode_mascara_removida.tif', erode_image_sem_mascara)
    # recuperação da imagem por meio da imagem original e imagem anterior como máscara
    mascara_final = np.zeros(erode_image_sem_mascara.shape[:2], dtype='uint8')
    height, width = mascara_final.shape
    upper_left = (width // 2, height // 1000)
    bottom_right = (width * 3 // 1000, height * 3 // 2)
    cv.rectangle(mascara_final, upper_left, bottom_right, (33, 33, 33), thickness=-1)
    imagem_final = cv.bitwise_or(imagem, imagem, mask=mascara_final)
    cv.imwrite('imagens/saida/cookies_mask_to_original.tif', imagem_final)


def quest3():
    return


def escolhe_questao():
    print('======== MENU DE OPÇÕES ========')
    print('1 - QUESTÃO 1')
    print('2 - QUESTÃO 2')
    print('3 - QUESTÃO 3')
    print('4 - Sair')
    opcao = int(input('Escolha uma das opções acima: '))

    while opcao < 1 or opcao > 4:
        print('Opção Inválida!')
        print('======== MENU DE OPÇÕES ========')
        print('1 - QUESTÃO 1')
        print('2 - QUESTÃO 2')
        print('3 - QUESTÃO 3')
        print('4 - Sair')
        opcao = int(input('Escolha novamente uma das opções acima: '))

    if opcao == 1:
        quest1()
    elif opcao == 2:
        quest2()
    elif opcao == 3:
        quest3()
    else:
        print('Volte sempre!')


if __name__ == '__main__':
    quest2()
