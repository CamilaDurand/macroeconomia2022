#!/usr/bin/env python
# coding: utf-8

# # REPORTE 4

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

# Mendoza en el presente texto busca demostrar cómo los métodos tradicionales, optando particularmente por la opción teórica de keynes, siguen siendo sumamente útiles para los problemas macroeconómicos contemporáneos además de modelar la política monetaria no convencional, extendiendo el modelo estándar de demanda y oferta agregada de economía cerrada para incorporar las innovaciones de la política monetaria. El autor señala como antes de la crisis económica de 2008-2009 el principal instrumento de estados unidos en materia de política monetaria era la tasa de interés y cómo a raíz de la crisis se optó por dos instrumentos de política no convencionales: el anuncio sobre la trayectoria futura de la tasa de interés de corto plazo y la intervención  directa en el mercado de bonos de largo plazo. Esta situación de usar instrumentos no convencionales ya había sido prevista por Keynes. Para examinar esta situación y sus efectos el texto ejemplifica un modelo en el que la Fed administra la tasa  de interés de corto plazo, no la oferta monetaria, la cual es endógena y además añade un mercado de bonos de largo plazo al modelo IS-LM, en el que solo existe un mercado de bonos de corto plazo. 
# 
# El modelo ofrecido por Mendoza es de suma utilidad pues gracias a el esposible discutir los efectos sobre las variables endógenas de la política fiscal y de los choques adversos de oferta. Con ello, el texto sigue un enfoque particularizado en analizar la estructura del modelo y pautar su reacción ante diferentes sucesos, lo cual permite entender de manera clara la predictibilidad macroeconómica sugerida por el autor. Otra de las ventajas del texto es su carácter sumamente explicativo, pues no se limita a explicar los sucesos sino el cómo se llega allí y la composición de las variables, de manera intuitiva, matemática y gráfica. Sin embargo,  también posee características a mejorar en futuras investigaciones  como el abordar los modelos contemporáneos, su fortalezas y debilidades para contrastarlo con  los modelos tradicionales que usa en su análisis.
# 
# La contribución de Mendoza es clara, pues permite entender cómo los métodos y modelos antiguos siguen ayudando a la comprensión de los problemas macroeconómicos más preocupantes de nuestro tiempo. El modelo que elabora genera una gran amplitud de entendimiento al respecto de la expansión cuantitativa. Asimismo el enfoque explicativo usado en el texto permite que personas sin mucho conocimiento económico pueda entender las políticas monetarias no convencionales y sus efectos a través de un lenguaje sencillo. Y al ser un modelo teórico tan bien explicado puede ser aplicado por muchos analistas como punto de referencia en un futuro cercano. 
# 
# Para seguir adelante con lo ya investigado por Mendoza resulta clave ver la aplicación de este modelo en contextos más actuales y realidades más cercanas como la peruana durante los últimos años, tomando en cuenta los préstamos y depósitos bancarios más comunes en los mercados financieros de la región como señala Manuelito. Asimismo sería oportuno seguir profundizando en la flexibilización cuantitativa y ver de manera directa la aplicación de instrumentos no convencionales en economías con características similares como Macas y Zhangallimbay lo hicieron sobre Ecuador.
# 
# 
# Manuelito, S. (2014). Los mercados financieros en América Latina y el financiamiento de la inversión: hechos estilizados y propuestas para una estrategia de desarrollo. Recuperado 24 de septiembre de 2022, de https://repositorio.cepal.org/handle/11362/5337
# 
# MACAS, G. O., & ZHANGALLIMBAY, D. J. (2019). Evaluación de impacto de los instrumentos de política monetaria no convencional en la liquidez de la economía: la experiencia ecuatoriana. Espacios, 40, 28. 

# ### CODIGO EN PYTHON

# Encuentre las ecuaciones de Ingreso $(Y^e)$  y tasa de interes $(r^e)$ de equilibrio(Escriba paso a paso la derivacion de estas ecuaciones).

# Se tiene a la función IS:
# $$r = \frac{β_o}{h} - \frac{β_1}{h}Y$$
# 
# donde: 
# 
# $$β_o = C_o - I_o + G_o + X_o$$    
# $$y$$    
# $$β_1 = 1 - (b - m)(1 - t)$$
# 
# Además, se tiene a la función LM:
# $$r = - \frac{1}{j} \frac{M_o^s}{P_o} + \frac{k}{j}Y$$

# ##### Derivación de la ecuación de $Y^e$
# Como se tiene r en los dos modelos, entonces igualamos:
# 
# $$\frac{β_o}{h} - \frac{β_1}{h}Y = - \frac{1}{j} \frac{M_o^s}{P_o} + \frac{k}{j}Y$$
# 
# $$\frac{k}{j}Y + \frac{β_1}{h}Y = \frac{β_o}{h}+\frac{1}{j}\frac{M_0^s}{P_0}$$
# 
# $$Y(\frac{k}{j} + \frac{β_1}{h}) = \frac{β_o}{h}+\frac{1}{j}\frac{M_0^s}{P_0}$$
# 
# $$Y(P_0kh+jβ_1P_0) = β_ojP_o+M_0^sh$$
# 
# Despejamos Y:
# 
# $$Y^e = \frac{jβ_oP_0+M_0^sh}{P_0kh+β_1jP_0}$$
# 
# $$Y^e = \frac{jβ_oP_0}{P_0kh+β_1jP_0}+\frac{M_0^sh}{P_0kh+β_1jP_0}$$
# 
# $$Y^e = \frac{P_0(jβ_o)}{P_0(kh+β_1j)}+\frac{M_0^sh}{P_0(kh+β_1j)}$$
# 
# Para que la ecuación sea útil en nuestro modelo, la acomodamos:
# 
# $$Y^e = \frac{jβ_o}{kh+jB_1}+(\frac{h}{kh+jβ_1})\frac{M_0^s}{P_0}$$

# ##### Derivación de la ecuación de interés $r^e$
# Como ya se tiene Y^e, entonces reemplazamos en la ecuación de r:
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y$$
# 
# $$ r^e = \frac{B_0}{h} - \frac{B_1}{h}Y^e$$
# 
# $$ r^e = \frac{B_0}{h} - \frac{B_1}{h} [\frac{jB_0}{kh+jB_1}+(\frac{h}{kh+jB_1})\frac{M_0^s}{P_0}]$$
# 
# $$ r^e = \frac{B_0}{h} - \frac{jB_0B_1}{h(kh+jB_1)} + \frac{B_1}{h}(\frac{h}{kh+jB_1})\frac{M_0^s}{P_0}$$
# 
# $$ r^e = \frac{B_0P_0(kh+jB_1)-jB_0B_1P_0+hB_1M_0^s}{hP_o(kh+jB_1)}$$
# 
# $$ r^e = \frac{khB_0P_0+jB_0B_1P_0-jB_0B_1P_0+hB_1M_0^s}{hP_0(kh+jB_1)}$$
# 
# $$ r^e = \frac{khB_0P_0+hB_1M_0^s}{hP_0(kh+jB_1)}$$
# 
# $$ r^e = \frac{khB_0P_0}{hP_0(kh+jB_1)}+\frac{hB_1M_0^s}{hP_0(kh+jB_1)}$$
# 
# Para que la ecuación sea útil en nuestro modelo, la acodamos:
# 
# $$r^e = \frac{kB_0}{kh + jB_1} - (\frac{B_1}{kh + jB_1})\frac{M^s_0}{P_0}$$

# Grafique el equilibrio simultáneo en los mercados de bienes y de dinero.

# In[4]:


# Curva IS

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
   r_IS = (Co + Io + Go + Xo)/h - ( ( 1-(b-m)*(1-t) ) / h)*Y  
   return r_IS

r_is = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
   # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def r_LM(k, j, Ms, P, Y):
   r_LM = - (1/j)*(Ms/P) + (k/j)*Y
   return r_LM

r_lm = r_LM( k, j, Ms, P, Y)


# In[5]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(r_lm)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
# Curva IS
ax.plot(Y, r_is, label = "IS", color = "#A1C298") #IS
# Curva LM
ax.plot(Y, r_lm, label="LM", color = "#FA7070")  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
# Graficar la linea horizontal - r
plt.axvline(x=51.5,  ymin= 0, ymax= 0.52, linestyle = ":", color = "#FFAE6D")
# Grafica la linea vertical - Y
plt.axhline(y=93, xmin= 0, xmax= 0.52, linestyle = ":", color = "#FFAE6D")

# Plotear los textos 
plt.text(49,100, '$E_0$', fontsize = 14, color = '#E3C770')
plt.text(0,100, '$r_0$', fontsize = 12, color = '#E3C770')
plt.text(53,-10, '$Y_0$', fontsize = 12, color = '#E3C770')

# Título, ejes y leyenda
ax.set(title="Modelo IS-LM", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# #### Estatica comparativa

# Analice los efectos sobre las variables endógenas Y, r de una disminución del gasto fiscal. $∆Go<0$ . El análisis debe ser intuitivo, matemático y gráfico.

# ##### Intuitivamente: 
# Mercado de Bienes : 
# $$G_o↓ → DA↓ → DA < Y → Y↓$$
# 
# Mercado de dinero
# $$Y↓ → M_d↓ → M_d < M_s → r↓$$

# ##### Análisis matemático: 

# In[6]:


# Se nombra a las variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# Se nombra a las variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Se especifica qué son Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Se pone a las ecuaciones del producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
Y_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
r_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[7]:


df_Y_eq_Go = diff(Y_eq, Go)
print("El Diferencial del Producto con respecto al diferencial del gasto autonomo = ", df_Y_eq_Go)


# In[8]:


df_r_eq_Go = diff(r_eq, Go)
print("El Diferencial de la tasa de interes con respecto al diferencial del gasto autonomo = ", df_r_eq_Go)


# ##### Graficando:

# In[9]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 60
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
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[10]:


# NUEVA curva IS: reducción Gasto de Gobierno (Go)
   
# Definir SOLO el parámetro cambiado
Go = 20

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
   r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
   return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[11]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS_(G_0)", color = "#9C2C77") #IS_orginal
ax.plot(Y, r_G, label = "IS_(G_1)", color = "#9C2C77", linestyle = 'dashed') #IS_modificada

ax.plot(Y, i, label="LM", color = "#CD104D")  #LM_original

# Texto y figuras agregadas
plt.axvline(x=40,  ymin= 0, ymax= 0.42, linestyle = ":", color = "grey")
plt.axhline(y=70, xmin= 0, xmax= 0.40, linestyle = ":", color = "grey")

plt.axvline(x=55.5,  ymin= 0, ymax= 0.55, linestyle = ":", color = "grey")
plt.axhline(y=100, xmin= 0, xmax= 0.55, linestyle = ":", color = "grey")

plt.text(60,100, '$IS G_O$', fontsize = 14, color = 'black')
plt.text(45,70, '$IS G_1$', fontsize = 14, color = 'black')

plt.text(-1,102, '$r_0$', fontsize = 12, color = '#FD841F')
plt.text(-1,72, '$r_1$', fontsize = 12, color = '#FD841F')
plt.text(55.5,-5, '$Y_0$', fontsize = 12, color = '#FD841F')
#plt.text(50,52, '$E_1$', fontsize = 14, color = '#3D59AB')
plt.text(40,-5, '$Y_1$', fontsize = 12, color = '#FD841F')

plt.text(65, 60, '↙', fontsize=25, color='#FD841F')

# Título, ejes y leyenda
ax.set(title="Politica Fiscal Contractiva", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# Analice los efectos sobre las variables endógenas Y, r de una disminución de la masa monetaria.$∆M^s<0$ . El análisis debe ser intuitivo, matemático y gráfico.

# **Intuitivamente**
# - Mercado de dinero
# $$ Ms↓ → M^o↓ → M^o < M^d → r↑ $$
# 
# - Mercado de Bienes
# $$ r↑ → I↓ → DA<Y → Y↓ $$ 

# **Matematicamente**

# In[12]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
Y_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
r_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[13]:


df_r_eq_Ms = diff(r_eq, Ms)
print("El Diferencial de la tasa de interes con respecto al diferencial de la masa monetaria = ", df_r_eq_Ms) 


# In[14]:


df_Y_eq_Ms = diff(Y_eq, Ms)
print("El Diferencial del producto con respecto al diferencial de la masa monetaria = ", df_Y_eq_Ms)  


# El diferencial del Gasto autónomo (ΔG_o) afecta tanto a la Curva IS como a la curva LM y es una política fiscal contractiva.
# 
# El diferencial del Gasto autónomo (ΔG_o) afecta tanto a la Curva IS como a la curva LM y es una política fiscal contractiva. En este sentido, lo que muestran las ecuaciones es que $Y_e$ y r son negativas por lo que la curva IS se desplazará a la izquierda.

# **graficando**

# In[15]:


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
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[16]:


# Definir SOLO el parámetro cambiado
Ms = 200

# Generar nueva curva LM con la variacion del Ms
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[17]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "C1") #IS_orginal
ax.plot(Y, i, label="LM_(MS_0)", color = "C0")  #LM_original

ax.plot(Y, i_Ms, label="LM_(MS_1)", color = "C0", linestyle = 'dashed')  #LM_modificada

# Lineas de equilibrio_0 
plt.axvline(x=51.5,  ymin= 0, ymax= 0.57, linestyle = ":", color = "grey")
plt.axhline(y=93, xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")

# Lineas de equilibrio_1 
plt.axvline(x=56,  ymin= 0, ymax= 0.55, linestyle = ":", color = "grey")
plt.axhline(y=85, xmin= 0, xmax= 0.6, linestyle = ":", color = "grey")
plt.text(58,87, '$E_0$', fontsize = 14, color = 'black')

#plt.axhline(y=68, xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")

# Textos ploteados
plt.text(49,100, '$E_1$', fontsize = 14, color = 'black')
plt.text(-1,80, '$r_0$', fontsize = 12, color = 'black')
plt.text(53,-40, '$Y_0$', fontsize = 12, color = 'black')
#plt.text(50,52, '$E_1$', fontsize = 14, color = '#3D59AB')
#plt.text(-1,72, '$r_1$', fontsize = 12, color = '#3D59AB')
#plt.text(47,-40, '$Y_1$', fontsize = 12, color = '#3D59AB')

#plt.text(69, 115, '→', fontsize=15, color='grey')
#plt.text(69, 52, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title="Efecto de una disminucion de la masa monetaria", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# Analice los efectos sobre las variables endógenas Y, r de un incremento de la tasa de impuestos. $∆t>0$ . El análisis debe ser intuitivo, matemático y gráfico.

# **Intuitivamente**
# 
# - Mercado de Bienes
# $$ t↑ → Co↓ → DA↑ → DA<Y → Y↓ $$ 
# 
# - Mercado de dinero
# $$ Y↓ → M_d↓ → M_d < M_s → r↓  $$ 
# 

# **MATEMATICAMENTE**

# In[18]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
Y_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
r_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[19]:


df_r_eq_t = diff(r_eq, t)
print("El Diferencial de la tasa de interés con respecto al diferencial de la tasa impositiva = ", df_r_eq_t)


# In[20]:


df_Y_eq_t = diff(Y_eq, t)
print("El Diferencial del producto con respecto al diferencial de la tasa impositiva = ", df_Y_eq_t)


# **GRAFICO**

# In[21]:


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


# In[22]:


# Definir SOLO el parámetro cambiado
t = 3

# Generar nueva curva IS con la variacion del t
def r_IS_t(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_t = r_IS_t (b, m, t, Co, Io, Go, Xo, h, Y)


# In[23]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "C1") #IS_orginal
ax.plot(Y, i, label="LM_(MS_0)", color = "C0")  #LM_original

ax.plot(Y, r_t, label="IS_(t_1)", color = "C1", linestyle = 'dashed')  #LM_modificada

# Lineas de equilibrio_0 
plt.axvline(x=53,  ymin= 0, ymax= 0.55, linestyle = ":", color = "grey")
plt.axhline(y=80, xmin= 0, xmax= 0.55, linestyle = ":", color = "grey")

# Lineas de equilibrio_1 
plt.axvline(x=57,  ymin= 0, ymax= 0.57, linestyle = ":", color = "grey")
plt.axhline(y=90, xmin= 0, xmax= 0.55, linestyle = ":", color = "grey")
plt.text(54,87, '$E_0$', fontsize = 14, color = 'black')

#plt.axhline(y=68, xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")

# Textos ploteados
plt.text(48,75, '$E_1$', fontsize = 14, color = 'black')
plt.text(-1,92, '$r_0$', fontsize = 12, color = 'black')
plt.text(55,-40, '$Y_0$', fontsize = 12, color = 'black')
#plt.text(50,52, '$E_1$', fontsize = 14, color = '#3D59AB')
#plt.text(-1,72, '$r_1$', fontsize = 12, color = '#3D59AB')
#plt.text(47,-40, '$Y_1$', fontsize = 12, color = '#3D59AB')

#plt.text(69, 115, '→', fontsize=15, color='grey')
#plt.text(69, 52, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title="Efecto de un incremento de la tasa de impuestos", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# In[ ]:




