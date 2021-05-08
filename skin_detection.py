import numpy as np
import math
import cv2 as cv


def skin_detection(imagem):
    nova_imagem = np.copy(imagem)
    nova_imagem = np.float32(nova_imagem) / 255

    R = nova_imagem[:, :, 0]
    G = nova_imagem[:, :, 1]
    B = nova_imagem[:, :, 2]
    result = B
    for i in range(0, len(nova_imagem)):
        for j in range(0, len(nova_imagem[0])):
            r = R[i][j] / R[i][j] + G[i][j] + B[i][j]
            g = G[i][j] / R[i][j] + G[i][j] + B[i][j]

            # limite superior
            F1 = -1.367 * math.pow(r, 2) + 1.0743 * r + 0.2
            # limite inferior
            F2 = -0.776 * math.pow(r, 2) + 0.5601 * r + 0.18

            w = math.pow((r - 0.33), 2) + math.pow((g - 0.33), 2) > 0.001

            up = (R[i][j] - ((G[i][j] / 2 + B[i][j] / 2)))

            down = math.sqrt((R[i][j] - G[i][j]) ** 2 + (R[i][j] - B[i][j]) * (G[i][j] - B[i][j]))

            teta = math.acos(up / down)

            if B[i][j] > G[i][j]:
                H = 360.0 - teta
            H = teta

            if np.all(np.logical_and((np.logical_and((np.logical_and((g < F1), (g > F2))), (w > 0.001))),
                                     (np.logical_or((H > 240), (H <= 20))))) == True:
                skin = 1
            else:
                skin = 0
            result[i][j] = H

    return (result * 255).astype(np.uint8)
