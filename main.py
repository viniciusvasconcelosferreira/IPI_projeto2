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

"""
Primeiro passo (Detecção de pele)
 -  Normalização do modelo RGB

"""


def boolstr_to_floatstr(x):
    if x is True:
        return 1
    elif x is False:
        return 0
    else:
        return x


def quest1_2():
    imagem = cv.imread('imagens/entrada/morf_test.png')
    if cp.RGB_color_space(imagem) and cp.HSV_color_space(imagem):
        print('Arquivos gerados com sucesso!')
    else:
        print('Erro ao gerar arquivos!')


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
    cv.imshow("Original / Binarizada", np.hstack([imagem, bin1]))
    cv.waitKey(0)
    plt.figure(figsize=(12, 8))
    plt.suptitle('Original')
    plt.imshow(imagem, cmap='gray')
    plt.show()
    plt.suptitle('Binarizada')
    plt.imshow(bin1, cmap='gray')
    plt.show()


def quest2():
    return


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
    quest1_2()
