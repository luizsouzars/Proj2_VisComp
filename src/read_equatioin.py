"""Imports"""
import cv2
from matplotlib import pyplot as plt

#  Le imagem do arquivo
img_rgb = cv2.imread(r"data\Emc2.png", cv2.IMREAD_COLOR)

# Transforma em escala de cinza
gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Binariza a imagem
ret, thresh = cv2.threshold(gray, 125, 255, 0)

# Template
template = cv2.imread(r"data\Template_E.png", cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

img = img_rgb.copy()
method = eval("cv2.TM_CCOEFF")

# Parametros: frame a ser detectado, frame principal e método
res = cv2.matchTemplate(template, thresh, method)

# Retorna valores de coordenadas
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc

bottom_right = (top_left[0] + w, top_left[1] + h)
# Baseada nas coordenadas de retorno, o objeto detectado é desenhado
cv2.rectangle(thresh, top_left, bottom_right, 0, 2)
cv2.rectangle(img, top_left, bottom_right, 255, 2)

# Plot
plt.subplot(121)
plt.imshow(img, "gray")
plt.subplot(122)
plt.imshow(thresh, "gray")
plt.show()


# '''Mostra a imagem'''
# cv2.imshow('Binary Image',thresh)

# '''Aguarda que alguma tecla seja teclada'''
# cv2.waitKey(0)

# '''Fecha a janela'''
# cv2.destroyAllWindows()
