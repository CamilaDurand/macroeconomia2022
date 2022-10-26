#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from causalgraphicalmodels import CausalGraphicalModel


# ## MODELO INGRESO-GASTO: CURVA IS 

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


# ## **REPORTE DEL TEXTO** ##

# El working paper llamado “the case of peru” escrito por Martinelli y vega aborda la inflación crónica del Perú durante las décadas de 1970 y 1980. El texto examina la evolución de estos ciclos y las respuestas que recibió por parte del gobierno y sus entidades a través de los años, haciendo principal hincapié en que dicha inflación fue el resultado de la necesidad de tributación inflacionaria en un régimen de predominio fiscal de la política monetaria. Los autores buscan entender los ciclos de inflación, hiperinflación y difusión en el contexto particularizado de Perú así como examinar a fondo las políticas monetarias y fiscales que se elaboraron durante los años de estudio. Asimismo analiza el cambio de la opinión pública luego de las consecuencias terribles que tuvieron las políticas económicas previas, lo cual puede explicar la credibilidad del régimen político de 1990. Todo ello dando énfasis a los actores intervinientes como el gobierno, el banco central, la instituciones internacionales como el FMI y la sociedad representada en la opinión pública.
# 
# El presente texto presenta múltiples fortalezas al igual que debilidades. Iniciando con las fortalezas es importante destacar es análisis multidimensional que realizan, pues no solo se detienen en estadísticas, predictores y factores meramente económicos , sino que en su análisis también están muy presentes factores políticos y sociales que influyen en las respuestas hacia la inflación. Junto a esta fortaleza se encuentra su análisis temporal pues al examinar más de una década logra entender y comparar en sus diversos contextos el funcionamiento de ciertas políticas fiscales y monetarias. Y para culminar con sus fortalezas más resaltantes está la relevancia que le da al marco teórico pues explica el entendimiento general de conceptos como inflación, pero asimismo los debate y contextualiza al caso peruano. Ahora bien, es importante señalar alguna de sus debilidades o mejor dichos aspectos complementarios con los que se enriquecería mucho, que en mi opinión es un mayor análisis comparado entre diferentes países de la región latinoamericana. Si bien el texto menciona casos como el de Argentina, lo hace muy brevemente y de manera pautada, para entender la complejidad de los ciclos inflacionarios es necesario también ver el panorama completo y explorar con mayor amplitud las respuestas generadas por diversos países de nuestra región.
# 
# Considero que la contribución del texto es amplia, ya que no solo aborda los ciclos de inflación vividos en el perú sino que explica alguna de sus causales y las  respuestas que obtuvo durante los años. Su análisis es bastante particularizado al contexto peruano y entiende como factores políticos , sociales e incluso climáticos tiene grandes impactos en la economía peruana. Asimismo, ayuda a señalar los errores de las políticas económicas previas, dicho énfasis permite distinguir entre opciones que se deben de tratar de no repetir jamás y algunas otras que pueden servir mesuradamente en situaciones particulares. El abordaje del presente texto aplana bastante el terreno para otros investigadores en este ámbito y plantea incluso nuevas preguntas en torno a los factores no económicos  abordados y el impacto en la economía nacional que pueden tener. 
# 
# Finalmente, es importante señalar que aunque el texto realiza múltiples avances en el campo de la investigación y análisis económico puede ser complementado de manera realmente fructífera en otros trabajos. Basándome en lo mencionado previamente sobre lo beneficioso que sería comparar la situación peruana con más casos de la región latinoamericana sería bueno seguir a autores como Rubio u otros autores que brinden propuestas conceptuales para análisis compartidos. De esta manera sería posible establecer de manera clara los factores comunes y diferenciales que han significado las mejores prácticas en materia económica y visualizar cómo podrían ser replicadas en contextos como el peruano. Asimismo trabajos como el de Leal, Molina y Zilberman podrían inspirar a establecer proyecciones de inflación con métodos de machine learning en nuestro país. El paper analizado brinda muchas luces sobre la situación atravesada por nuestro país en torno a la inflación sin embargo, aún hay ciertas mejoras metodológicas como las dos mencionadas que podrían tener un gran impacto en los estudios económicos peruanos y regionales.
# 
# 
# **Bibliografía**
# 
# Rubio, P. (2007). Aportes a una propuesta conceptual para alcance y medición común en proyectos de simplificación en la región (No. publication-detail, 7101. html? id= 7118). Inter-American Development Bank.
# 
# Leal, F., Molina, C., & Zilberman, E. (2020). Proyección de la inflación en Chile con métodos de machine learning. Banco Central de Chile.
# 

# ## **CODIGO EN PYTHON**

# ## a. El modelo ingeso-gasto: la curva IS

# $$Y^e= DA$$
# $$DA= (Co+Io+Go+Xo-hr)+[(b-m)*(1-t)]$$
# $$Y^e= \frac{1}{1-(b-m)(1-t)}*(Co+Io+Go+Xo-hr)$$
# 
# 
# $$ Y = C + I + G + X - M $$
# $$ Y - T = C + I - T + G + X - M $$
# 
# $$ Y^d = C + I - T + G + X - M $$
# 
# Ello se pude representar en:
# $$ (Y^d - C) + (T - G) + (M - X) = I $$
# 

# Reemplazando
# $$ (Y^d - C_0 - bY^d) + (T - G_0) + (mY^d - X_0) = I_0 - hr $$

# Teniendo en cuenta la condicion de equilibrio de Y
# $$ [1 - (b - m)(1 - t)]Y - (C_0 + G_0 + X_0) = I_0 - hr $$

# Entonces podemos obtener la curva IS en funcion de la tasa de interes
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$

# **usando la funcion de equilibrio IS donde r esta en funcion de Y, encuetro los diferenciales de r Y para observar la pendiente**

# La funcion de equilibrio IS donde r esta en funcion de Y es
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$
# Asumiendo que hay una reduccion en la tasa de interes y por tanto desplazamiento de la curva 
# $$ r = \frac{r_0-r_1}{Y_0-Y_1} = \frac{(-)}{(+)} = (-) $$
# Entonces
# $$ r = \frac{∆r}{∆Y}<0$$

# **Lea la sección 4.4 del material de enseñanza y explique cómo se deriva la curva IS a partir del equilibrio Y=DA y grafique:**

# DERIVACION
# 
# equilibrio $(Y = DA)$:
# $$ Y = \frac{1}{1 - (b - m)(1 - t)} (C_0 + I_0 + G_0 + X_0 - hr) $$
# Ello tambien puede estar en funcion de la tasa de interes
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$
# Si es que queremos simplificar 
# $ B_0 = C_0 + G_0 + I_0 + X_0  $  y $  B_1 = 1 - (b - m)(1 - t) $ 
# 
# entonces simplificando quedaria: \$$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$

# **DEMANDA AGREGADA**

# In[4]:


Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)

#--------------------------------------------------
# Recta de 45°

a = 2.5 

def L_45(a, Y):
    L_45 = a*Y
    return L_45

L_45 = L_45(a, Y)

#--------------------------------------------------
# Segunda curva de ingreso de equilibrio

    # Definir cualquier parámetro autónomo
Go = 35

# Generar la ecuación con el nuevo parámetro
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_G = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


# **CURVA IS**

# In[5]:


Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[6]:


fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: ingreso de Equilibrio
ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.plot(Y, DA_IS_K, label = "DA_0", color = "C0") 
ax1.plot(Y, DA_G, label = "DA_1", color = "C0") 
ax1.plot(Y, L_45, color = "#404040") 

ax1.axvline(x = 70.5, ymin= 0, ymax = 0.69, linestyle = ":", color = "grey")
ax1.axvline(x = 54,  ymin= 0, ymax = 0.54, linestyle = ":", color = "grey")

ax1.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
ax1.text(2.5, -3, '$◝$', fontsize = 30, color = 'black')
ax1.text(72, 0, '$Y_1$', fontsize = 12, color = 'black')
ax1.text(56, 0, '$Y_0$', fontsize = 12, color = 'black')
ax1.text(70, 170, 'DA(r1); siendo r_1< r_0', fontsize = 12, color = 'black')
ax1.text(67, 185, 'E_1', fontsize = 12, color = 'black')
ax1.text(50, 142, 'E_0', fontsize = 12, color = 'black')
ax1.text(67, 130, 'DA(r_0)', fontsize = 12, color = 'black')


ax1.set(title = "Derivación de la curva IS a partir del equilibrio $Y=DA$", xlabel = r'Y', ylabel = r'DA')
ax1.legend()

#SEGUNDO GRAFICO
ax2.yaxis.set_major_locator(plt.NullLocator())   
ax2.xaxis.set_major_locator(plt.NullLocator())

ax2.plot(Y, r, label = "IS", color = "C1") 

ax2.axvline(x = 70.5, ymin= 0, ymax = 1, linestyle = ":", color = "grey")
ax2.axvline(x = 54,  ymin= 0, ymax = 1, linestyle = ":", color = "grey")
plt.axhline(y = 151.5, xmin= 0, xmax = 0.7, linestyle = ":", color = "grey")
plt.axhline(y = 165, xmin= 0, xmax = 0.55, linestyle = ":", color = "grey")

ax2.text(72, 128, '$Y_1$', fontsize = 12, color = 'black')
ax2.text(56, 128, '$Y_0$', fontsize = 12, color = 'black')
ax2.text(1, 153, '$r_1$', fontsize = 12, color = 'black')
ax2.text(1, 167, '$r_0$', fontsize = 12, color = 'black')
ax2.text(72, 152, 'E_1', fontsize = 12, color = 'black')
ax2.text(55, 166, 'E_0', fontsize = 12, color = 'black')

ax2.legend()

plt.show()


# ## **b. La Curva IS o el equilibrio Ahorro- Inversión**

# **DERIVACION**
# 
# Para llegar al equilibrio Ahorro-Inversion  restamos la tributacion de las partes de la igualdad
# 
# $$ Y - T = C + I - T + G + X - M $$
# 
# $$ Y^d = C + I - T + G + X - M $$
# 
# Ello se puede expresar en:
# 
# $$ (Y^d - C) + (T - G) + (M - X) = I $$
# 
#  Entonces las tres partes de la derecha con los 3 componenetes del ahorro total $(S)$ que son: ahorro privado $(S_p)$, ahorro del gobierno $(S_g)$ y ahorro esterno $(S_e)$
#  
#  De ahi observqamos que el ahorro total es igual a la inversion
#  
#  $$ S_p + S_g + S_e = I $$
# 
# $$ S(Y) = I(r) $$
# Hciendo los reemplazos respectivos
# 
# $$[Y^d-Co-bY^d)]+[T-Go]+[mY^d-Xo]=I_0 - hr$$
# 
# Tomando en cuenta la condicion de equilibrio de Y y sus equivalencias:
# $$ [1 - (b - m)(1 - t)]Y - (C_0 + G_0 + X_0) = I_0 - hr $$
# 
# La curva IS entonces puede expresarse con una ecuacion donde la tasa de interes es una funcion del ingreso
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$

# **GRAFICO**

# In[7]:


Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Equation 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (1/h)*(Co + Io + Go + Xo) - (1/h)*(1-(b-m)*(1-t))*Y
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[8]:


# Dimensions
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curves to plot
ax.plot(Y, r, label = "IS", color = "#4fa167") #Demanda agregada

# Remove the quantities from the axes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

ax.text(100, 130, 'IS', fontsize = 12, color = 'black')

# Title, axes adn legend
ax.set(title = "Curva IS de equilibrio en el mercado de bienes" , xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# ## **c. Desequilibrios en el mercado de bienes**

# In[9]:


Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3
Y = np.arange(Y_size)
# Equation 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (1/h)*(Co + Io + Go + Xo) - (1/h)*(1-(b-m)*(1-t))*Y
    return r_IS
r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)

# Dimensions
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curves to plot
ax.plot(Y, r, label = "IS", color = "#4fa167") #Demanda agregada

# Remove the quantities from the axes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())
ax.text(-10, 180, '$r_A$', fontsize = 12, color = 'black')
ax.text(22, 180, 'C', fontsize = 12, color = 'black')
ax.text(37, 180, 'A', fontsize = 12, color = 'black')
ax.text(52, 180, 'B', fontsize = 12, color = 'black')
ax.text(22, 127, '$Y_C$', fontsize = 12, color = 'black')
ax.text(37, 127, '$Y_A$', fontsize = 12, color = 'black')
ax.text(52, 127, '$Y_B$', fontsize = 12, color = 'black')
#/// Lineas horizontales | Lineas verticales
plt.axvline(x = 22,  ymin= 0, ymax= 0.69, linestyle = ":", color = "grey")
plt.axvline(x = 37,  ymin= 0, ymax= 0.69, linestyle = ":", color = "grey")
plt.axvline(x = 52,  ymin= 0, ymax= 0.69, linestyle = ":", color = "grey")
plt.axhline(y = 180, xmin= 0, xmax = 0.7, linestyle = ":", color = "grey")
#// Texto
plt.text(24, 160, 'Exceso de', fontsize = 11.5, color = '#3D59AB')
plt.text(24, 155, 'demanda', fontsize = 11.5, color = '#3D59AB')
#// Texto
plt.text(65, 190, 'Exceso de', fontsize = 11.5, color = '#3D59AB')
plt.text(65, 185, 'Oferta', fontsize = 11.5, color = '#3D59AB')

# Title, axes adn legend
ax.set(title = " Desequilibrios en el Mercado de Bienes", xlabel= 'Y', ylabel= 'r')
ax.legend()

ax.legend()
plt.show()


# Como sabemos la curva IS representa los puntos de equilibrio entre el ahorro y la inversion, es decir el mecado de bienes. Los puntos fuera de la curva indica desquilibrio en el mercado debido a diversos factores como la sobreproduccion. Los desequilibrios puede ser tanto del lado de la oferta como de la demanda, ambos por excesos. El punto A del grafico es el punto de equilibrio entre el ahorro y la inversion, mientras que el punto B a la derecha de la curva muestra desequilibrio pues pese a mantener la tasa interes y tener la inversion constante, el ahorro es mayor que el del punto A, ya que el ingreso en B es mas alto, por tanto B representa excesos de oferta. Finalmente vemos el punto C a la izquierda de la curva representando el exceso de demanda, pues igualmente mantiene la tasa de interes pero el ahorro es menor que en A debudo a que su ingreso es menor. 

# ## **d. Movimientos de la curva IS**

# **Politica fiscal contractiva con caida del fasto de gobierno**

# Analisis intuitivo:
# segun la ecuacion
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$
# 
# Al ser $\frac{1}{h}(C_0 + G_0 + I_0 + X_0)$ el intercepto 
# y al ser $\frac{1 - (b - m)(1 - t)}{h}$ la pendiente de la curva IS
# 
# $$∆Go=G$$
# $$G0↓ → G↓ → DA↓ → DA<Y → Y↓$$

# In[10]:


# Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # NUEVA curva IS
    
Go = 50

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[11]:


# Dimensiones del gráfico
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "black") #IS orginal
ax.plot(Y, r_G, label = "IS_G", color = "C1", linestyle = 'dashed') #Nueva IS

# Texto agregado
plt.text(47, 162, '∆Go', fontsize=12, color='black')
plt.text(49, 159, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title = "Incremento en el Gasto de Gobierno $(G_0)$", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# **Política Fiscal Expansiva con una caída de la Tasa de Impuestos**

# Analisis intuitivo:
# 
# $$t↓→Co↑→DA↑→DA>Y→Y↑$$

# In[12]:


# Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.7

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # NUEVA curva IS
    
t = 0.3

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_t = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[13]:


y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "black") #IS orginal
ax.plot(Y, r_t, label = "IS_t", color = "C1", linestyle = 'dashed') #Nueva IS

# Texto agregado
plt.text(47, 162, '∆t', fontsize=12, color='black')
plt.text(47, 158, '→', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title = "Disminucion en la Tasa de Interés $(t)$", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# **caída de la Propensión Marginal a Consumir**

# Analisis intuitivo
# $$b↓→Co↓ → DA↓→DA<Y→Y↓$$

# In[14]:


# Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.7

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # NUEVA curva IS
    
b = 0.3

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_t = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[15]:


y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "black") #IS orginal
ax.plot(Y, r_t, label = "IS_b", color = "C1", linestyle = 'dashed') #Nueva IS

# Texto agregado
plt.text(60, 133, '∆b', fontsize=12, color='black')
plt.text(60, 129, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title = " caída de la Propensión Marginal a Consumir $(b)$", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()

