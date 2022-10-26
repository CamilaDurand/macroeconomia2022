#!/usr/bin/env python
# coding: utf-8

# # MODELO DA-OA

# Camila Durand- 20200918 Con Luciana Sarmiento- 20202422 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from causalgraphicalmodels import CausalGraphicalModel


# In[2]:


import matplotlib.pyplot as plt
import sympy as sy
import math
import sklearn
from sympy import *


# In[3]:


import scipy as sp
import networkx
import ipywidgets as widgets
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col


# ### REPORTE DE LA LECTURA

# El texto escrito por Dancourt analiza las medidas tomadas por el banco central peruano para preservar la estabilidad macroeconómica a principios de los 2000s, particularmente la implementación de un sistema de metas de inflación y la acumulación de suficientes reservas de divisas. Teniendo lo ya expresado como objetivo el texto analiza los principales instrumentos de política monetaria, como la tasa de interés a corto plazo fijada por el banco central o tasa de interés de referencia, el coeficiente de encaje legal en moneda nacional y extranjera y la intervención esterilizada en el mercado de divisas, al igual que el procesos de crédito bancario de (des)dolarización en el periodo del 2002 al 2013. El texto además de analizar el rol de las políticas monetarias aplicadas también analiza el contexto externo que fue excepcionalmente favorable en el marco temporal analizado. De esta manera, Dancourt busca enfocar su análisis en el efecto concreto de la política monetaria en la actividad económica y el nivel de precios a través de el canal del crédito y el canal del tipo de cambio, los más importantes en la economía peruana. 
# 
# Las fortalezas del enfoque usado por Dancourt es que al analizar el efecto del contexto internacional puede particularizar el efecto de las políticas monetarias previamente descritas. Asimismo, logra insertar estudios de otros autores en su examinación de manera complementaria lo que le ayuda a alcanzar una amyor profundidad. También cabe recalcar que otra de las fortalezas del texto es que prevé posibles debilidades y las trata de subsanar ampliando su campo de análisis y realizando las preguntas correctas como ¿por qué no hubo una gran recesión o crisis bancaria en 2008-2009 como la de 1998-2000? Lo que permite visualizar el efecto concreto de ciertas variables. Pese a estas fortalezas y a sus avances por subsanar sus debilidades el texto podría avanzar mucho màs si es que incorpora ciertos mecanismos comparativos con las políticas monetarias de otros países en contextos similares en los que se cais en una gran dependencia de los precios internacionales de ciertos productos como los metales.
# 
# La contribución de Dancourt recae en las múltiples perspectivas que resalta, pues no solo se enfoca en el proceso temporal, sino también en los mecanismos y sus efectos. Logrando explicar mediante modelos de múltiples autores tales como taylor o Blanchard el efecto de las políticas monetarias y en qué circunstancias son más útiles ciertos mecanismos de transmisión de la política monetaria. Dependiendo de en cuál de las 3 etapas se encuentra la economía dancourt señala cómo actúan los diferentes factores exogenos y endogenos como la expansión del crédito en soles durante la segunda y tercera etapa es el desarrollo del mercado local de bonos públicos, seguido luego por los bonos privados y corporativos denominados en moneda local. Asimismo durante todo el texto explica detalladamente las gráficas y las ecuaciones que llevaron a ello. Finalmente, la contribución mas importante en mi opinión es la mirada a futuro que da y las recomendaciones que realizó para evitar crisis bancarias usando las desdolarización y políticas monetarias como la planteada por blanchard. 
# 
# Para seguir avanzando en esta materia sería conveniente realizar un seguimiento y evaluar situaciones en las que sus recomendaciones han sido aplicadas, examinando textos como los de Girón. Además como se mencionó previamente sería de gran ayuda ofrecer una perspectiva comparada con casos similares en la región latinoamericana, para examinar y distinguir de manera los efectos del contexto y las políticas monetarias como tal siguiendo a autores como del Río Rivera. 
# 
# del Río Rivera, M. A., & Kuscevic, C. M. M. (2014). Desdolarización financiera en Bolivia. Estudios económicos, 3-25.
# 
# Girón, A. (2006). Financiamiento del desarrollo: endeudamiento externo y reformas financieras.
# 

# ### CODIGO EN PYTHON

# #### A partir del siguiente sistema de ecuaciones que representan el modelo DA-OA

# #### 1. Encuentre las ecuaciones de Ingreso ($Y^e$) y tasa de interés ($r^e$) de equilibrio (Escriba paso a paso la derivación de estas ecuaciones).

# Se tiene para la DA que:
# 
# $$ P = \frac{hMo^s + jB_o}{h} - \frac{jB_1 + hk}{h}Y . . . (1) $$
# 
# 
# Se tiene que OA:
# $$ P = P^e + θ(Y - \bar{Y}) $$ 

# ##### Derivando la ecuación de Ingreso ($Y^e$)

# Para poder encontrar el $Y^e$ se tiene que igualar las ecuaciones de la DA y la OA por su común $P^e$.
# 
# $$\frac{hM_o^s+jB_o}{h} - \frac{jB_1+hk}{h}Y^e_{DA-OA} = P^e + θ(Y^e_{DA-OA} - \bar{Y})$$
# 
# $$\frac{hM_o^s+jB_o}{h} - \frac{jB_1+hk}{h}Y^e_{DA-OA} = P^e + θY^e_{DA-OA} - θ\bar{Y}$$
# 
# $$- \frac{jB_1+hk}{h}Y^e_{DA-OA} - θY^e_{DA-OA} = P^e - θ\bar{Y} - \frac{hM_o^s+jB_o}{h}$$
# 
# $$-Y^e_{DA-OA} (θ + \frac{jB_1+hk}{h}) = P^e - θ\bar{Y} - \frac{hM_o^s+jB_o}{h} $$
# 
# $$Y^e_{DA-OA} (θ + \frac{jB_1+hk}{h}) = - P^e + θ\bar{Y} + \frac{hM_o^s+jB_o}{h} $$
# 
# $$Y^e_{DA-OA} = [\frac {1}{(θ + \frac{jB_1+hk}{h})}][\frac{hM_o^s+jB_o}{h}- P^e + θ\bar{Y}]$$
# 
# Finalmente, para hallar $P^e$, reemplazamos a $Y^e$ en la ecuación de la OA:
# 
# $$P^e = P^e + θ(Y^e - \bar{Y})$$
# 
# $$P^e = P^e + θ( [ \frac{1}{(θ + \frac{jB_1 + hk}{h})} ][(\frac{h Mo^s + jB_o}{h} - P^e + θ\bar{Y})] - \bar{Y} )$$

# ##### Derivando la ecuación de interés ($r^e$)

# Para poder encontrar el $r^e$ se reemplaza a $Y^e$ en el modelo $IS-LM$
# 
# $$ r = \frac{B_o}{h} - \frac{B_1}{h}Y$$
# 
# $$ r^e = \frac{B_o}{h} - \frac{B_1}{h}Y^e$$
# 
# $$ r^e = \frac{B_o}{h} - \frac{B_1}{h} [\frac{jB_o}{kh+jB_1}+(\frac{h}{kh+jB_1})\frac{M_o^s}{P}]$$
# 
# $$ r^e = \frac{B_o}{h} - \frac{jB_oB_1}{h(kh+jB_1)} + \frac{B_1}{h}(\frac{h}{kh+jB_1})\frac{M_o^s}{P}$$
# 
# $$ r^e = \frac{B_oP(kh+jB_1)-jB_oB_1P+hB_1M_o^s}{hP(kh+jB_1)}$$
# 
# 
# $$ r^e = \frac{khB_oP+hB_1M_o^s}{hP(kh+jB_1)}$$
# 
# $$ r^e = \frac{khB_oP}{hP(kh+jB_1)}+\frac{hB_1M_o^s}{hP(kh+jB_1)}$$
# 
# $$r^e = \frac{kB_o}{kh + jB_1} - (\frac{B_1}{kh + jB_1})\frac{M^s_o}{P}$$
# 
# Si:
# 
# $$\frac{M_o^s}{P} = M_0^s-P$$
# 
# Entonces y finalmente:
# 
# $$r^e = \frac{kB_o}{kh + jB_1} - (\frac{B_1}{kh + jB_1})(M^s_o-P)$$
# 
# $$r^e = \frac{kB_o}{kh + jB_1} - (\frac{B_1}{kh + jB_1})(M^s_0-P^e)$$
# 
# $$r^e = \frac{kB_o}{kh + jB_1} - (\frac{B_1}{kh + jB_1}) (M^s_0-P^e + θ( [ \frac{1}{(θ + \frac{jB_1 + hk}{h})}] [(\frac{h Mo^s + jB_o}{h} - P^e + θ\bar{Y})] - \bar{Y} ))$$

# #### 2. Grafique el equilibrio simultáneo en el modelo DA-OA.

# In[4]:


#1--------------------------
    # Demanda Agregada
    
# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.4
m = 0.5
t = 0.8

k = 2
j = 1                
Ms = 200             
P  = 8  

Y = np.arange(Y_size)


# Ecuación

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_AD = P_AD(h, Ms, j, B0, B1, k, Y)


#2--------------------------
    # Oferta Agregada
    
# Parámetros

Y_size = 100

Pe = 100 
θ = 3
_Y = 20   

Y = np.arange(Y_size)


# Ecuación

def P_AS(Pe, _Y, Y, θ):
    P_AS = Pe + θ*(Y-_Y)
    return P_AS

P_AS = P_AS(Pe, _Y, Y, θ)


# In[5]:


# líneas punteadas autómaticas

    # definir la función line_intersection
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

    # coordenadas de las curvas (x,y)
A = [P_AD[0], Y[0]] # DA, coordenada inicio
B = [P_AD[-1], Y[-1]] # DA, coordenada fin

C = [P_AS[0], Y[0]] # L_45, coordenada inicio
D = [P_AS[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección

intersec = line_intersection((A, B), (C, D))
intersec # (y,x)


# In[6]:


# Gráfico del modelo DA-OA

# Dimensiones del gráfico
y_max = np.max(P)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar 
ax.plot(Y, P_AD, label = "DA", color = "#FF577F") #DA
ax.plot(Y, P_AS, label = "OA", color = "#FF884B") #OA

# Líneas punteadas
plt.axhline(y=intersec[0], xmin= 0, xmax= 0.5, linestyle = ":", color = "grey")
plt.axvline(x=intersec[1],  ymin= 0, ymax= 0.49, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 200, '$P_0$', fontsize = 12, color = '#251B37')
plt.text(53, 25, '$Y_0$', fontsize = 12, color = '#251B37')
plt.text(50, 202, '$E_0$', fontsize = 12, color = '#251B37')


# Eliminar valores de ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Título, ejes y leyenda
ax.set(title="DA-OA", xlabel= r'Y', ylabel= r'P')
ax.legend()

plt.show()


# ### Estatica comparativa

# #### 1. Analice los efectos sobre las variables endógenas P y r de una disminución del gasto fiscal.($△G_o < 0$). El análisis debe ser intuitivo, matemático y gráfico. En una figura, se debe que usar los ejes r e Y (modelo IS-LM), y en la otra, los ejes P y r (modelo DA-OA).

# **Intuitivamente**
# 
# Modelo IS-LM:
# 
# $$Go↓ → DA↓ → DA < Y → Y↓$$
# 
# $$Y↓ → M^d↓ → M^d < M^s → r↓$$
# 
# Modelo DA-OA:
# 
# $$Y↓ → θ(Y-\bar{Y})↓ → P↓$$

# **Matematicamente**

# In[7]:


#Variables de la curva IS
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

#Variables de la curva LM 
k, j, Ms, P, Y = symbols('k j Ms P Y')

#Variables de OA
Pe, _Y, Y, θ = symbols('Pe, _Y, Y, θ')

#Betas
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

#Y de equilibrio DA-OA
Y_eq = ( (1)/(θ + ( (j*beta_1+h*k)/h) ) )*( ( (h*Ms+j*beta_0)/h ) - Pe + θ*_Y )

#P de equilibrio DA-OA 
P_eq = Pe + θ*(Y_eq - _Y)

#r de equilibrio DA-OA
r_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms - P_eq)


# In[8]:


# Efecto del cambio de Go sobre r en DA-OA
df_r_eq_Go = diff(r_eq, Go)
print("El diferencial de r con respecto al diferencial de Go:", df_r_eq_Go)
print("\n")

#Efecto del cambio de Go sobre P en DA-OA
df_P_eq_Go = diff(P_eq, Go)
print("El diferencial de P con respecto al diferencial de Go es:", df_P_eq_Go) 


# El diff de $r$ con el diferencial de $G_o$ es negativo.
# 
# El diferencial de $P$ con el diferencial de $G_o$ es negativo

# **Graficamente**

# In[9]:


# IS-LM

#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 55
Xo = 2
h = 0.8
b = 0.4
m = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 8           

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva IS: disminución del gasto Fiscal (Go)

# Definir SOLO el parámetro cambiado
Go = 20

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_Go = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)

#DA-OA

#1--------------------------
    # Demanda Agregada ORGINAL
    
# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.4
m = 0.5
t = 0.8

k = 2
j = 1                
Ms = 200             
P  = 8  

Y = np.arange(Y_size)


# Ecuación

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_AD = P_AD(h, Ms, j, B0, B1, k, Y)

#--------------------------------------------------
    # NUEVA Demanda Agregada

# Definir SOLO el parámetro cambiado

Go = 13

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

# Generar la ecuación con el nuevo parámetro

def P_AD_Go(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_Go = P_AD_Go(h, Ms, j, B0, B1, k, Y)


#2--------------------------
    # Oferta Agregada ORIGINAL
    
# Parámetros

Y_size = 100

Pe = 70
θ = 3
_Y = 20  

Y = np.arange(Y_size)

# Ecuación

def P_AS(Pe, _Y, Y, θ):
    P_AS = Pe + θ*(Y-_Y)
    return P_AS

P_AS = P_AS(Pe, _Y, Y, θ)

# Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: IS-LM
# Gráfico IS-LM
    
ax1.plot(Y, r, label = "$IS$", color = "#06283D") 
ax1.plot(Y, r_Go, label="$IS$", color = "#06283D", linestyle ='dashed')  
ax1.plot(Y, i, label="$LM$", color = "#256D85")  

ax1.axvline(x=45,  ymin= 0, ymax= 0.45, linestyle = ":", color = "grey")
ax1.axvline(x=58,  ymin= 0, ymax= 0.58, linestyle = ":", color = "grey")
ax1.axhline(y=64,  xmin= 0, xmax= 0.45, linestyle = ":", color = "grey")
ax1.axhline(y=91,  xmin= 0, xmax= 0.57, linestyle = ":", color = "grey")

ax1.text(35, 95, '↙', fontsize=20, color='blue')
ax1.text(58, 0, '$Y_0$', fontsize=12, color='#47B5FF')
ax1.text(45, 0, '$Y_1$', fontsize=12, color='#47B5FF')
ax1.text(0, 91, '$r_0$', fontsize=12, color='#47B5FF')
ax1.text(0, 63, '$r_1$', fontsize=12, color='#47B5FF')

ax1.set(title="Efectos de  $G_0 < 0$ en $IS-LM$", xlabel= r'Y', ylabel= r'r')
ax1.legend()


#Gráfico DA-OA

ax2.plot(Y, P_AD, label = "$DA$", color = "#7895B2") 
ax2.plot(Y, P_Go, label = "$DA_{\Delta G_0 < 0}$", color = "#7895B2", linestyle = 'dashed')
ax2.plot(Y, P_AS, label = "$OA$", color = "#829460") 

ax2.axvline(x=56,  ymin= 0, ymax= 0.52, linestyle = ":", color = "#A1C298")
ax2.axvline(x=48.5,  ymin= 0, ymax= 0.45, linestyle = ":", color = "#A1C298")
ax2.axhline(y=179,  xmin= 0, xmax= 0.56, linestyle = ":", color = "#A1C298")
ax2.axhline(y=154,  xmin= 0, xmax= 0.49, linestyle = ":", color = "#A1C298")

ax2.text(36, 208, '↙', fontsize=20, color='#A1C298')

ax2.text(57.5, 0, '$Y_0$', fontsize=12, color='#B6E388')
ax2.text(45, 0, '$Y_1$', fontsize=12, color='#B6E388')
ax2.text(0, 146, '$P_1$', fontsize=12, color='#B6E388')
ax2.text(0, 180, '$P_0$', fontsize=12, color='#B6E388')

ax2.set(title="Efectos de $G_0 < 0$ en $DA-OA$", xlabel= r'Y', ylabel= r'P')
ax2.legend()

plt.show


# 2. **Analice los efectos sobre las variables endógenas Y, P y r de una disminución de la masa monetaria.**

# **Intuitivamente**
# 
# Modelo IS-LM: 
# $$ Ms↓ → M^s↓ → M^s < M^d → r↑ $$
# $$ r↑ → I↓ → DA↓ → DA < Y → Y↓ $$
# 
# Modelo DA-OA: 
# $$ Y↓ → θ(Y-\bar{Y})↓ → P↓$$

# **Matematicamente**

# In[10]:


# nombrar variables como símbolos de IS
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos de LM 
k, j, Ms, P, Y = symbols('k j Ms P Y')

# nombrar variables como símbolos para curva de oferta agregada
Pe, _Y, Y, θ = symbols('Pe, _Y, Y, θ')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio en el modelo DA-OA
Y_eq = ( (1)/(θ + ( (j*beta_1+h*k)/h) ) )*( ( (h*Ms+j*beta_0)/h ) - Pe + θ*_Y )

# Precio de equilibrio en el modelo DA-OA 
P_eq = Pe + θ*(Y_eq - _Y)

# Tasa de interés de equilibrio en el modelo DA-OA
r_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms - P_eq)
#((h*Ms+j*beta_0)/h) - ((j*beta_1+h*r)/h)*((P-Pe-θ*_Y)/θ)


# In[11]:


# Efecto del cambio de Precio esperado sobre Tasa de Interés en el modelo DA-OA
df_Y_eq_Ms = diff(Y_eq, Ms)
print("El Diferencial del Producto con respecto al diferencial de la masa monetaria =", df_Y_eq_Ms)
print("\n")

# Efecto del cambio de Precio esperado sobre Tasa de Interés en el modelo DA-OA
df_r_eq_Ms = diff(r_eq, Ms)
print("El Diferencial de la tasa de interés con respecto al diferencial de la masa monetaria = ", df_r_eq_Ms)
print("\n")

# Efecto del cambio de Precio esperado sobre Tasa de Interés en el modelo DA-OA
df_P_eq_Ms = diff(P_eq, Ms)
print("El Diferencial del nivel de precios con respecto al diferencial de la masa monetaria =", df_P_eq_Ms)


# **Graficamente:**

# In[12]:


# IS-LM

#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.4
m = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)
#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 500             
P  = 8           

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM: incremento en la Masa Monetaria (Ms)

# Definir SOLO el parámetro cambiado
Ms = 200

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)

#DA-OA
#1--------------------------
    # Demanda Agregada ORGINAL
    
# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.4
m = 0.5
t = 0.8

k = 2
j = 1                
Ms = 400             
P  = 8  

Y = np.arange(Y_size)


# Ecuación

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_AD = P_AD(h, Ms, j, B0, B1, k, Y)

#--------------------------------------------------
    # NUEVA Demanda Agregada

# Definir SOLO el parámetro cambiado

Ms = 200

# Generar la ecuación con el nuevo parámetro

def P_AD_Ms(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_Ms = P_AD_Ms(h, Ms, j, B0, B1, k, Y)


#2--------------------------
    # Oferta Agregada ORIGINAL
    
# Parámetros

Y_size = 100

Pe = 70
θ = 3
_Y = 20  

Y = np.arange(Y_size)

# Ecuación

def P_AS(Pe, _Y, Y, θ):
    P_AS = Pe + θ*(Y-_Y)
    return P_AS

P_AS = P_AS(Pe, _Y, Y, θ)
# Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: IS-LM
    
ax1.plot(Y, r, label = "IS", color = "C1") #IS
ax1.plot(Y, i, label="LM", color = "C0")  #LM
ax1.plot(Y, i_Ms, label="LM_Ms", color = "C0", linestyle ='dashed')  #LM

ax1.axvline(x=67.5,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax1.axvline(x=56,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax1.axhline(y=72,  xmin= 0, xmax= 0.66, linestyle = ":", color = "grey")
ax1.axhline(y=87,  xmin= 0, xmax= 0.56, linestyle = ":", color = "grey")

ax1.text(75, 110, '∆$M_s$', fontsize=12, color='black')
ax1.text(76, 102, '←', fontsize=15, color='grey')
ax1.text(60, -60, '←', fontsize=15, color='grey')
ax1.text(0, 77, '↑', fontsize=15, color='grey')
ax1.text(50, -65, '$Y_1$', fontsize=12, color='C0')
ax1.text(70, -65, '$Y_0$', fontsize=12, color='black')
ax1.text(0, 91, '$r_1$', fontsize=12, color='C0')
ax1.text(0, 63, '$r_0$', fontsize=12, color='black')


ax1.set(title="Efectos de una reduccion en la masa monetaria", xlabel= r'Y', ylabel= r'r')
ax1.legend()


#---------------------------------
    # Gráfico 2: DA-OA

ax2.plot(Y, P_AD, label = "AD", color = "C4") #DA
ax2.plot(Y, P_Ms, label = "AD_Ms", color = "C4", linestyle = 'dashed') #DA_Ms
ax2.plot(Y, P_AS, label = "AS", color = "C8") #OA

ax2.axvline(x=87.5,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax2.axvline(x=56,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax2.axhline(y=270,  xmin= 0, xmax= 0.87, linestyle = ":", color = "grey")
ax2.axhline(y=175,  xmin= 0, xmax= 0.56, linestyle = ":", color = "grey")

ax2.text(70, 30, '←', fontsize=15, color='grey')
ax2.text(36, 300, '←', fontsize=15, color='grey')
ax2.text(35, 310, '∆$M_s$', fontsize=12, color='black')
ax2.text(0, 230, '↓', fontsize=15, color='grey')

ax2.text(58, 0, '$Y_1$', fontsize=12, color='C4')
ax2.text(90, 0, '$Y_0$', fontsize=12, color='black')
ax2.text(0, 158, '$P_1$', fontsize=12, color='C4')
ax2.text(0, 270, '$P_0$', fontsize=12, color='black')

ax2.set(xlabel= r'Y', ylabel= r'P')
ax2.legend()

plt.show


# 3. **Analice los efectos sobre las variables endógenas Y, P y r de un incremento de la tasa de impuestos**

# **Intuitivamente**
# 
# Modelo IS-LM: 
# $$ t↑ → Co↓ → DA↑ → DA<Y → Y↓ $$ 
# $$ Y↓ → M_d↓ → M_d < M_s → r↓  $$ 
# 
# Modelo DA-OA: 
# $$ Y↓ → θ(Y-\bar{Y})↓ → P↓$$

# **Matematicamente**
# 
# IS-LM 
# $${ΔY_e}=\frac{j}{kh+ Δt} + \frac{h}{kh + Δt} Δt>0$$
# 
# $$ΔY= (-) + (-)  = (-) < 0$$
# 
# $${Δr}=\frac{k}{kh+ Δt} + \frac{Δt}{kh + Δt} Δt> 0$$
# 
# $$Δr= (-) + (-)  = (-) < 0$$
# 
#  DA-OA:
# 
# $$ ΔP = P^e + θ(YΔ - \bar{Y})  Δt> 0 $$
# 
# $$ ΔP= (+) + (+) * (-) = (-) < 0$$

# **Graficamente** 

# In[13]:


# IS-LM

#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)
#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 500             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)
#--------------------------------------------------
    # NUEVA curva LM: incremento en tasa impositiva (t)

# Definir SOLO el parámetro cambiado
t = 3

# Generar nueva curva IS con la variacion del t
def r_IS_t(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_t = r_IS_t (b, m, t, Co, Io, Go, Xo, h, Y)
    

#DA-OA
#1--------------------------
    # Demanda Agregada ORGINAL
    
# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

k = 2
j = 1                
Ms = 400             
P  = 8  

Y = np.arange(Y_size)


# Ecuación

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_AD = P_AD(h, Ms, j, B0, B1, k, Y)

#--------------------------------------------------
    # NUEVA Demanda Agregada

# Definir SOLO el parámetro cambiado

t = 8
B1 = 1 - (b-m)*(1-t)

# Generar la ecuación con el nuevo parámetro

def P_AD_t(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_t = P_AD_t(h, Ms, j, B0, B1, k, Y)


#2--------------------------
    # Oferta Agregada ORIGINAL
    
# Parámetros

Y_size = 100

Pe = 70
θ = 3
_Y = 20  

Y = np.arange(Y_size)

# Ecuación

def P_AS(Pe, _Y, Y, θ):
    P_AS = Pe + θ*(Y-_Y)
    return P_AS

P_AS = P_AS(Pe, _Y, Y, θ)
# Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: IS-LM
    
ax1.plot(Y, r, label = "IS", color = "C1") #IS
ax1.plot(Y, i, label="LM", color = "C0")  #LM
ax1.plot(Y, r_t, label="IS_(t_1)", color = "C1", linestyle = 'dashed')  #LM_modificada

ax1.axvline(x=53,  ymin= 0, ymax= 0.55, linestyle = ":", color = "grey")
ax1.axvline(x=57,  ymin= 0, ymax= 0.57, linestyle = ":", color = "grey")
ax1.axhline(y=80,  xmin= 0, xmax= 0.55, linestyle = ":", color = "grey")
ax1.axhline(y=90,  xmin= 0, xmax= 0.55, linestyle = ":", color = "grey")

ax1.text(70, 100, '∆$t$', fontsize=12, color='black')
ax1.text(70, 94, '←', fontsize=15, color='grey')
ax1.text(60, -60, '←', fontsize=15, color='grey')
ax1.text(0, 77, '↓', fontsize=15, color='grey')
ax1.text(50, -65, '$Y_1$', fontsize=12, color='C0')
ax1.text(70, -65, '$Y_0$', fontsize=12, color='black')
ax1.text(0, 91, '$r_1$', fontsize=12, color='C0')
ax1.text(-1,92, '$r_0$', fontsize = 12, color = 'black')


ax1.set(title="Efectos de un aumento en la tasa impositiva", xlabel= r'Y', ylabel= r'r')
ax1.legend()



#---------------------------------
    # Gráfico 2: DA-OA

ax2.plot(Y, P_AD, label = "AD", color = "C4") #DA
ax2.plot(Y, P_t, label = "AD_t", color = "C4", linestyle = 'dashed') #DA_t
ax2.plot(Y, P_AS, label = "AS", color = "C8") #OA

ax2.axvline(x=87.5,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax2.axvline(x=78,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax2.axhline(y=275,  xmin= 0, xmax= 0.87, linestyle = ":", color = "grey")
ax2.axhline(y=240,  xmin= 0, xmax= 0.75, linestyle = ":", color = "grey")

ax2.text(80, 30, '←', fontsize=15, color='grey')
ax2.text(70, 295, '←', fontsize=15, color='grey')
ax2.text(70, 307, '∆$t$', fontsize=12, color='black')
ax2.text(0, 240, '↓', fontsize=15, color='grey')

ax2.text(67, 0, '$Y_1$', fontsize=12, color='C4')
ax2.text(90, 0, '$Y_0$', fontsize=12, color='black')
ax2.text(0, 200, '$P_1$', fontsize=12, color='C4')
ax2.text(0, 270, '$P_0$', fontsize=12, color='black')

ax2.set(xlabel= r'Y', ylabel= r'P')
ax2.legend()

plt.show


# In[ ]:





# In[ ]:




