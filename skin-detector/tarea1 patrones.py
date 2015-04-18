# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import misc
from scipy import linalg
import math

# <codecell>

piel_mu = 16*[0]
piel_sigma = 16*[0]
piel_w = 16*[0]
nopiel_mu = 16*[0]
nopiel_sigma = 16*[0]
nopiel_w = 16*[0]

# <codecell>

piel_mu[0] =  np.array((73.53, 29.94, 17.76))
piel_mu[1] = np.array((249.71, 233.94, 217.49))
piel_mu[2] = np.array((161.68, 116.25, 96.95))
piel_mu[3] = np.array((186.07, 136.62, 114.40))
piel_mu[4] = np.array((189.26, 98.37, 51.18))
piel_mu[5] = np.array((247.00, 152.20, 90.84))
piel_mu[6] = np.array((150.10, 72.66, 37.76))
piel_mu[7] = np.array((206.85, 171.09, 156.34))
piel_mu[8] = np.array((212.78, 152.82, 120.04))
piel_mu[9] = np.array((234.87, 175.43, 138.94))
piel_mu[10] = np.array((151.19, 97.74, 74.59))
piel_mu[11] = np.array((120.52, 77.55, 59.82))
piel_mu[12] = np.array((192.20, 119.62, 82.32))
piel_mu[13] = np.array((214.29, 136.08, 87.24))
piel_mu[14] = np.array((99.57, 54.33, 38.06))
piel_mu[15] = np.array((238.88, 203.08, 176.91))

piel_sigma[0] = np.diag((765.40, 121.44, 112.80))
piel_sigma[1] = np.diag((39.94, 154.44, 396.05))
piel_sigma[2] = np.diag((291.03, 60.48, 162.85))
piel_sigma[3] = np.diag((274.95, 64.60, 198.27))
piel_sigma[4] = np.diag((633.18, 222.40, 250.69))
piel_sigma[5] = np.diag((65.23, 691.53, 609.92))
piel_sigma[6] = np.diag((408.63, 200.77, 257.57))
piel_sigma[7] = np.diag((530.08, 155.08, 572.79))
piel_sigma[8] = np.diag((160.57, 84.52, 243.90))
piel_sigma[9] = np.diag((163.80, 121.57, 279.22))
piel_sigma[10] = np.diag((425.40, 73.56, 175.11))
piel_sigma[11] = np.diag((330.45, 70.34, 151.82))
piel_sigma[12] = np.diag((152.76, 92.14, 259.15))
piel_sigma[13] = np.diag((204.90, 140.17, 270.19))
piel_sigma[14] = np.diag((448.13, 90.18, 151.29)) 
piel_sigma[15] = np.diag((178.38, 156.27, 404.99))

piel_w[0] = 0.0294
piel_w[1] = 0.0331
piel_w[2] = 0.0654
piel_w[3] = 0.0756
piel_w[4] = 0.0554
piel_w[5] =  0.0314
piel_w[6] = 0.0454
piel_w[7] = 0.0469
piel_w[8] = 0.0956
piel_w[9] = 0.0763
piel_w[10] = 0.1100
piel_w[11] = 0.0676
piel_w[12] = 0.0755
piel_w[13] = 0.0500
piel_w[14] = 0.0667
piel_w[15] = 0.0749


nopiel_mu[0] =  np.array((254.37, 254.41, 253.82))
nopiel_mu[1] = np.array((9.39, 8.09, 8.52))
nopiel_mu[2] = np.array((96.57, 96.95, 91.53))
nopiel_mu[3] = np.array((160.44, 162.49, 159.06))
nopiel_mu[4] = np.array((74.98, 63.23, 46.33))
nopiel_mu[5] = np.array((121.83, 60.88, 18.31))
nopiel_mu[6] = np.array((202.18, 154.88, 91.04))
nopiel_mu[7] = np.array((193.06, 201.93, 206.55))
nopiel_mu[8] = np.array((51.88, 57.14, 61.55))
nopiel_mu[9] = np.array((30.88, 26.84, 25.32))
nopiel_mu[10] = np.array((44.97, 85.96, 131.95))
nopiel_mu[11] = np.array((236.02, 236.27, 230.70))
nopiel_mu[12] = np.array((207.86, 191.20, 164.12))
nopiel_mu[13] = np.array((99.83, 148.11, 188.17))
nopiel_mu[14] = np.array((135.06, 131.92, 123.10))
nopiel_mu[15] = np.array((135.96, 103.89, 66.88))

nopiel_sigma[0] = np.diag((2.77, 2.81, 5.46))
nopiel_sigma[1] = np.diag((46.84, 33.59, 32.48))
nopiel_sigma[2] = np.diag((280.69, 156.79, 436.58))
nopiel_sigma[3] = np.diag((355.98, 115.89, 591.24))
nopiel_sigma[4] = np.diag((414.84, 245.95, 361.27))
nopiel_sigma[5] = np.diag((2502.24, 1383.53, 237.18))
nopiel_sigma[6] = np.diag((957.42, 1766.94, 1582.52))
nopiel_sigma[7] = np.diag((562.88, 190.23, 447.28))
nopiel_sigma[8] = np.diag((344.11, 191.77, 433.40))
nopiel_sigma[9] = np.diag((222.07, 118.65, 182.41))
nopiel_sigma[10] = np.diag((651.32, 840.52, 963.67))
nopiel_sigma[11] = np.diag((225.03, 117.29, 331.95))
nopiel_sigma[12] = np.diag((494.04, 237.69, 533.52))
nopiel_sigma[13] = np.diag((955.88, 654.95, 916.70))
nopiel_sigma[14] = np.diag((350.35, 130.30, 388.43))
nopiel_sigma[15] = np.diag((806.44, 642.20, 350.36))


nopiel_w[0] = 0.0637
nopiel_w[1] = 0.0516
nopiel_w[2] = 0.0864
nopiel_w[3] = 0.0636
nopiel_w[4] = 0.0747
nopiel_w[5] = 0.0365
nopiel_w[6] = 0.0349
nopiel_w[7] = 0.0649
nopiel_w[8] = 0.0656
nopiel_w[9] = 0.1189
nopiel_w[10] = 0.0362
nopiel_w[11] = 0.0849
nopiel_w[12] = 0.0368
nopiel_w[13] = 0.0389
nopiel_w[14] = 0.0943
nopiel_w[15] = 0.0477

# <codecell>

def binarizar(x):
    capa = np.mean(x, axis=2)
    return  capa>=128

# <codecell>

n_imagenes = 20
img = n_imagenes*[0]
img_mask = n_imagenes*[0]
for i in range(n_imagenes):
    j = i+1
    img[i] = misc.imread('' + str(j)+'.jpg')
    img_mask[i] = binarizar(misc.imread(''+str(j)+'_mask.jpg'))

# <codecell>

from scipy.stats import multivariate_normal
def P_piel(x):
    result = 0.0
    for i in range(16):
        result += piel_w[i]*multivariate_normal.pdf(x,mean=piel_mu[i],cov=piel_sigma[i])
    return result

def P_nopiel(x):
    result = 0.0
    for i in range(16):
        result += nopiel_w[i]*multivariate_normal.pdf(x,mean=nopiel_mu[i],cov=nopiel_sigma[i])
    return result

# <codecell>

##MUESTRA RESUMEN

# threshold = num_detecciones_correctas = 0.4
# num_falsos_positivos = 0
# num_falsos_negativos = 0
# num_positivos = 0
# num_negativos = 0
# total = 0

# for i in range(n_imagenes):
#     print('img'+str(i))
#     imagen = img[i]
#     mascara = img_mask[i]
#     prob_piel = P_piel(imagen)
#     prob_nopiel = P_nopiel(imagen)
#     print(prob_piel)
#     print(prob_nopiel)
#     cuociente = prob_piel/prob_nopiel
#     es_piel_prediccion = cuociente > threshold
#     detecciones_correctas = es_piel_prediccion == mascara
#     falsos_positivos = es_piel_prediccion & ~mascara
#     falsos_negativos = ~es_piel_prediccion & mascara
#     num_detecciones_correctas += detecciones_correctas.sum()
#     num_falsos_positivos += falsos_positivos.sum()
#     num_falsos_negativos += falsos_negativos.sum()
#     num_positivos += mascara.sum()
#     num_negativos += (~mascara).sum()
#     total += mascara.size
#     plt.subplot(211)
#     plt.imshow(mascara, cmap='Greys_r')
#     plt.subplot(212)
#     plt.imshow(es_piel_prediccion, cmap='Greys_r')
#     plt.show()
    

# precision = float(num_detecciones_correctas)/float(total)
# tasa_falsos_positivos = float(num_falsos_positivos)/float(num_negativos)
# tasa_falsos_negativos = float(num_falsos_negativos)/float(num_positivos)

# print("")
# print("THRESHOLD = "+str(threshold))
# print("detecciones correctas = " + str(num_detecciones_correctas))
# print("falsos posititvos = " +  str(num_falsos_positivos))
# print("falsos negativos = " +  str(num_falsos_negativos))
# print("tasa de falsos posititvos = " +  str(tasa_falsos_positivos))
# print("tasa de falsos negativos = " +  str(tasa_falsos_negativos))
# print("presicion = " + str(precision))
# print("pixeles totales = " +  str(total))

# <codecell>

num_positivos = 0
num_negativos = 0

from scipy.optimize import minimize

def diferencia_falsos_pos_neg(threshold):
    num_detecciones_correctas = 0
    num_falsos_positivos = 0
    num_falsos_negativos = 0
    num_positivos = 0
    num_negativos = 0
    total = 0
    
    for i in range(n_imagenes):
       # print("img" + str(i))
        imagen = img[i]
        mascara = img_mask[i]
        prob_piel = P_piel(imagen)
        prob_nopiel = P_nopiel(imagen)
        cuociente = prob_piel/prob_nopiel
        es_piel_prediccion = cuociente > threshold
        detecciones_correctas = es_piel_prediccion == mascara
        falsos_positivos = es_piel_prediccion & ~mascara
        falsos_negativos = ~es_piel_prediccion & mascara
        num_detecciones_correctas += detecciones_correctas.sum()
        num_falsos_positivos += falsos_positivos.sum()
        num_falsos_negativos += falsos_negativos.sum()
        num_positivos += mascara.sum()
        num_negativos += (~mascara).sum()
        total += mascara.size
        
    precision = float(num_detecciones_correctas)/float(total)
    tasa_falsos_positivos = float(num_falsos_positivos)/float(num_negativos)
    tasa_falsos_negativos = float(num_falsos_negativos)/float(num_positivos)
    diferencia = abs(tasa_falsos_positivos-tasa_falsos_negativos)
    
    print("")
    print("THRESHOLD = "+str(threshold))
    print("detecciones correctas = " + str(num_detecciones_correctas))
    print("falsos posititvos = " +  str(num_falsos_positivos))
    print("falsos negativos = " +  str(num_falsos_negativos))
    print("tasa de falsos posititvos = " +  str(tasa_falsos_positivos))
    print("tasa de falsos negativos = " +  str(tasa_falsos_negativos))
    print("diferencia de tasas = "+str(diferencia))
    print("presicion = " + str(precision))
    print("pixeles totales = " +  str(total))
    return abs(tasa_falsos_positivos-tasa_falsos_negativos)



def opt_binaria(inicio,fin,level, valor_inicio, valor_fin): 
    print("-----------------------")
    print("NIVEL "+str(level))
    media = float(inicio+fin)/2
    if (valor_inicio == None):
      valor_inicio = diferencia_falsos_pos_neg(inicio)
    if (valor_fin == None):
      valor_fin = diferencia_falsos_pos_neg(fin)
    dic = {valor_inicio:inicio, valor_fin:fin}
    minimo = min(dic)
    if(minimo < 0.001 or level==10):
        return dic[minimo]
    else:
        if(minimo == valor_inicio):
            return opt_binaria(inicio,media,level+1, valor_inicio, valor_fin)
        else:
            return opt_binaria(media,fin,level+1, valor_inicio, valor_fin)

# <codecell>

opt_binaria(0,3,0, None, None)

