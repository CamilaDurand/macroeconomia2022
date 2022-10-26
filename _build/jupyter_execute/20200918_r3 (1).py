#!/usr/bin/env python
# coding: utf-8

# ## *REPORTE 3*

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


# ## REPORTE DE LA LECTURA

# El texto de Mendoza, Mancilla y Velarde analiza el impacto de la pandemia en la economía peruana y la recuperación generalizada que vivió pese a la caída del pbi en 11% debido a la pandemia, la más severa atravesada por nuestro país desde 1989. Asimismo analiza el desplome de los ingresos de los trabajadores y como afecta la demanda y la producción de diferentes maneras, mostrando cómo el consumo se resiente y la demanda sufre un choque. Su análisis se basa en un modelos de cuarentena en una economía de dos sectores, en el que si bien ambos sufrieron los impactos de la pandemia, la contracción de la producción general fue desigual según la naturaleza de cada sector.
# 
# Una de las principales fortalezas del trabajo es que mediante la comparación de ambos sectores en el modelo se puede entender de manera realista el choque de la pandemia en la economía peruana. Asimismo el enfoque del texto permite desagregar los efectos tanto de consumidores como productores y otros actores en la economía peruana  que tienen expectativas racionales y con ello nos permite examinar sus implicaciones en el estado estacionario. El modelo explicativo usado permite entender simplificadamente cual fue el mejor curso de acción en ese contacto y como los créditos ayudaron a recuperar en gran parte la economía peruana. Sin embargo, al analizar los principales elementos de la demanda deja un vacío en el entendimiento completo de la situación pues no analiza aspectos de la oferta. 
# 
# El texto contribuye al entendimiento no solo de los efectos de la pandemia como tal, sino también ayuda a entender las opciones de respuesta como políticas monetarias. Asimismo, gracias al modelo utilizado es posible hacer comparaciones específicas entre la clase de efectos que tendrá la economía peruana en diferentes plazos de tiempo. Otro beneficio del modelo es que permite hacer comparaciones específicas y con ello entender que ambos sectores fueron afectados pero en diferente intensidad porque el sector 2 que continuó operando es afectado por la menor demanda proveniente del sector 1 que fue perjudicado por el cierre de actividades. Gracias a este conjunto de factores los lectores podemos entender particularmente mediante el segundo ejercicio como la política crediticia ayudan a tener un recuperación vigorosa haciendo que el PBI alcance sus niveles pre pandemia. 
# 
# Finalmente, como ya se mencionó y como él mismo texto reconoce uno de los principales pasos necesarios a tomar para ampliar el entendimiento del tema es incluir en el análisis los componentes de la oferta. Asimismo un siguiente paso a tomar sería desagregar los efectos de la pandemia no solo entre estos dos grandes sectores sino  también agregando factores relacionados a su estatus social como género, raza o grado de instrucción. Añadiendo variables como estas es posible rastrear del mejores maneras los efectos de la pandemia y usando investigaciones previas como las que el GRADE ofrece, al contar con trabajos como el de Rojas y Alvan. Finalmente podría ayudar al avance el realizar investigaciones similares comparando las respuestas que tuvieron otros países latinoamericanos, con el fin de examinar nuestra política crediticia más a fondo como ya se ha hecho en la región centroamericana por Moreno y Morales.
# 
# Moreno Brid, J. C., & Morales López, R. A. (2020). Centroamérica frente a la pandemia: retos de la política macroeconómica. Revista CEPAL-Edición Especial.
# 
# Rojas, V. & Alván, A. (2022). Madres jóvenes en pandemia: Una aproximación cualitativa a los retos del cuidado infantil [Análisis & Propuestas, 64]. Lima: GRADE; Niños del Milenio.
# 
# Rojas, V., Crivello, G. & Alván, A. (2022). Trayectorias educativas. Seguimiento de jóvenes peruanos en pandemia [Análisis & Propuestas, 63]. Lima: GRADE; Niños del Milenio.
# 

# ## **CODIGO EN PYTHON**

# 1. **Instrumentos de politica monetaria que usa en banco central**

# Se debe tomar en cuenta que, el Banco Central tiene la capacidad de controlar o gestionar la oferta monetaria y con esto puede hacer:
# - una política monetaria expansiva (incrementa la oferta de dinero y por ende, la demanda agregada, pero se reduce la tasa de interés) o 
# - una política monetaria contractiva (disminuye la oferta de dinero y aumenta la tasa de interés).
# 
# Y, para realizar estas acciones, el Banco Central tiene 3 posibles instrumentos:
# 
# 1. La oferta monetaria
# El Banco Central tiene la capacidad de regular el aumento o disminución de la oferta monetaria; y para esto, el Banco Central vende o compra activos financieros o bonos comerciales.
#    - En una política monetaria expansiva: se realiza compra de bonos del mercado -> se introduce dinero a la economía -> aumento de la oferta monetaria
#    - En una política monetaria contractiva:se realiza venta de bonos al mercado -> se retira dinero de la economía -> reducción de la oferta monetaria
#    
#    
# 2. El coeficiente legal de encaje
# El Banco Central puede "controlar" la porción de depósitos de un banco; y para esto, el Banco Central el banco realiza:
#    - En una política monetaria expansiva: se disminuye la tasa de encaje -> aumento del mutiplicador de dinero bancario -> aumento de posibilidad de creación de dinero bancario -> incremento de la cantidad de dinero bancario -> aumento de la oferta monetaria.
#    - En una política monetaria contractiva: se aumenta la tasa de encaje -> reducción del mutiplicador de dinero bancario -> disminución de posibilidad de creación de dinero bancario -> disminución de la cantidad de dinero bancario -> disminución de la oferta monetaria.
# 
# 
# 3. La tasa de interés
# Hace poco, el Banco Central se convirtió en un instrumento de política monetaria y se utiliza de la siguiente manera:
#    - En una política monetaria expansiva: El BC reduce su tasa de interés de referencia -> incremento de la cantidad de dinero prestada a los bancos -> aumento de la base monetaria -> aumento de la oferta monetaria
#    - En una política monetaria contractiva: El BC aumenta su tasa de interés de referencia -> disminución de dinero prestado a los bancos -> reducción de la base monetaria -> reducción de la oferta monetaria.

# 2. **Derive la oferta real de dinero y explique cada uno de sus componentes**
# #### Derivando:
# Se tiene de la oferta nominal es:
# $M_o^s$ 
# 
# Entonces, para que esta pase a ser la oferta real, se le debe ajustar al nivel de precios:
# $\frac {M_o^s}{P}$
# 
# 
# #### Explicando:
# Para la oferta real del dinero se tiene que:
# 
# $M_o^s$ representa a la cantidad de dinero circulando en la economía (variable exógena e instrumento de política); sin embargo, el dinero de los precios no es el mismo siempre -por la inflación-; por lo que este símbolo ($M_o^s$) será solo la masa monetaria nominal de dinero en la economía, no la oferta real.
# 
# Entonces, si se quiere pasar de la masa monetaria nominal (oferta nominal) a la oferta real, se debe ajustar ese valor por el nivel de los precios (p).
# $$\frac{M_o^s}{P}$$

# 3. **Derive la demanda real de dinero. Explique qué papel cumplen los parametros "k" y "j"**
# $$L=L_1+L_2$$
# $$L_1=kY$$
# $$L_2=-ji$$
# Entonces
# $$L=kY-ji$$
# 
# La previa derivación que corresponde a la demanda de dinero, la cual está relacionado a sus funciones. El primer bloque de la demanda (L1) está determinada por los motivos de transacción y precaución. Con respecto a las transacciones la magnitud de estas se encuentra en relación directa con el ingreso o producto de la economía. Y el motivo de precaución hace referencia a la capacidad de pago de deudas la cual depende directamente de sus ingresos. 
# 
# Es por ello que $L1=kY$ donde Y es el ingreso y “k” representa la sensibilidad o elasticidad de la demanda de dinero ante las variaciones de Y, al tener una relación positiva y directa ello indica que mientras más grande sea k mayor cantidad de dinero se va a demandar en cuanto se incremente el nivel de ingreso de la economía.
# 
# El segundo bloque de la demanda (L2) es determinada por el motivo de especulación el cual nos dice que se preferirá mantener liquidez en forma de dinero y no en forma de bonos cuando la tasa de interés se reduce pues ganaran menos si lo depositan en el banco, y lo contrario si aumenta. Es por ello que este segundo bloque depende negativa o inversamente de la tasa de interés de los bonos $L_2=-ji$. Donde el parámetro “j” indica cuán sensible es la demanda de dinero ante las variaciones de la tasa de interés nominal de los bonos, “i”.
# 

# 4. **Asumiendo que no hay inflación podemos asumir que i=r. Escriba en terminos reales la eucacion de equilibrio en el mercado de dinero**

# $$M^s=M^d$$
# $$\frac{M^s}{P}=kY-ji$$
# Y asumiendo que i=r
# $$M^s=P(kY-jr)$$

# 5.  **Grafique el equilibrio en el mercado de dinero**

# In[4]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[5]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#AF0171')

# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.5, "$r_0$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# ## Estatica comparativa en el Mercado de Dinero

# **Explique y grafique qué sucede en el mercado de dinero si ∆Y<0**

# Si el nivel del producto (Y) varia negativamente, es decir, se reduce entonces el intercepto se mueve hacia abajo.  La curva de demanda original se desplaza hacia abajo se llega a un nuevo punto de equilibrio que  muestra que la economia ha empeorado, la produccion se ha reducido y la gente demanda menos dinero, lo cual tomando en cuenta que el precio del dinero es la tasa de interes significa que la tasa de interes se reduce. la demanda de dinero se contrae y para la cantidad de dinero dada la tasa de inetres se reduce, la tasa de interes se comporta de manera endogena
# $$↓Y → ↓M_d → ↓r $$

# In[6]:


# Parameters con cambio en el nivel del producto
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y_1 = 20
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P, r, Y_1)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS

# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium, reduction of Y", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#AF0172')
#ax1.plot(MD_1, label= '$L_0$', color = '#AF0171')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.5, "$r_0$", fontsize = 12, color = 'black')
ax1.text(50,-3.5, "$(Ms/P)_0$", fontsize = 12, color = 'black')
ax1.text(50, 7.5, "$E_0$", fontsize = 12, color = 'black')

# Nuevas curvas a partir del cambio en el nivel del producto
ax1.plot(MD_1, label= '$L_1$', color = '#3D8361')
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=0.1, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")
ax1.text(0, 0.2, "$r_1$", fontsize = 12, color = 'black')
ax1.text(50, 0.2, "$E_1$", fontsize = 12, color = 'black')


ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# **Explique y grafique qué sucede en el mercado de dinero si ∆k<0**

# Si k que representa la elasticidad/sensibilidad d ela demanda de dinero ante la variacion en Y, entonces tambien se producira una reduccion en la demanda de dinero por lo que la curva se desplazara hacia abajo. ESta reduccion tambien produciria un desequilibrio y para reestablecer el equilibrio es necesario que la tasa de interes tambien se reduzca
# 
# $$↓k → ↓M_d → ↓r $$

# In[7]:


# Parameters con cambio en el nivel del producto
r_size = 100

k = 0.3
j = 0.2                
P  = 10 
Y= 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P, r, Y_1)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS

# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium, reduction of k", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#AF0172')
#ax1.plot(MD_1, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.5, "$r_0$", fontsize = 12, color = 'black')
ax1.text(50, 0, "$(Ms/P)_0$", fontsize = 12, color = 'black')
ax1.text(50, 7.5, "$E_0$", fontsize = 12, color = 'black')

# Nuevas curvas a partir del cambio en el nivel del producto
ax1.plot(MD_1, label= '$L_1$', color = '#3D8361')
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y= -4, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")
ax1.text(0, -3.5, "$r_1$", fontsize = 12, color = 'black')
ax1.text(50, -3.5, "$E_1$", fontsize = 12, color = 'black')


ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# **Explique y grafique qué sucede en el mercado de dinero si $∆M_s<0$**

# si la cantidad de dinero se reduce tambien lo hace la cantidad real de dinero porque esta en el numerador y guia por completo la direccion. La recta de la oferta de dinero se desplaza hacia la izquierda y por tanto se llega a un nuevo punto de equilibrio donde la tasa de interes aumenta y la demanda disminuye. La tasa de inetres ha aumenatdo con el proposito de reestablecer el equilibrio. 
# 
# Ello se puede evidenciar en:
# $$↓M_0→ ↓\frac{M^s}{P}$$
# $$↓\frac{M^s}{P}<M^d → ↑r$$

# In[8]:


# Parametros

r_size = 100

k = 0.6
j = 0.3                
P  = 10 
Y = 35

r = np.arange(r_size)

    # Ecuación

def Ms_MD(k, j, P, r, Y):
    Ms_MD = P*(k*Y - j*r)
    return Ms_MD

Ms_MD = Ms_MD(k, j, P, r, Y)


  # Equilibrio en el mercado de dinero

fig, ax1 = plt.subplots(figsize=(10, 8))

ax1.set(title="Money market equilibrium, reduction of $M_s$")
ax1.plot(Ms_MD, label= '$L_0$', color = '#CD5C5C')

ax1.axvline(x = 45,  ymin= 0, ymax= 1, color = "black")
ax1.axvline(x = 30,  ymin= 0, ymax= 1, color = "black",linestyle = 'dashed')
ax1.axhline(y=74, xmin= 0, xmax= 0.46, linestyle = ":", color = '#3D8361')
ax1.axhline(y=120, xmin= 0, xmax= 0.32, linestyle = ":", color = '#3D8361')
ax1.text(-10, 200, "$r$", fontsize = 14, color = 'black')
ax1.text(100, -120, "$M^s/M^d$", fontsize = 14, color = 'black')

ax1.text(-8, 75, "$r_0^e$", fontsize = 12, color = 'black')
ax1.text(-8, 120, "$r_1^e$", fontsize = 12, color = 'black')
ax1.text(41,-116, "$M_0^s/P_0$", fontsize = 12, color = 'black')
ax1.text(46,74, "$E_0$", fontsize = 10, color = 'black')
ax1.text(28, -116, "$M_1^s/P_0$", fontsize = 12, color = 'black')
ax1.text(31,120, "$E_1$", fontsize = 10, color = 'black')
ax1.text(99, -90, "$L(Y_0)$", fontsize = 10, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# ## CURVA LM

# 1. Derive paso a paso la curva LM matemáticamente (a partir del equilibrio en el Mercado Monetario) y grafique.
# 

# ##### Derivando:
# 
# Para hallar la curva LM a partir del equilibrio en el mercado monetario:
# 
# Se tiene previamente que en la demanda:
# 
# $$M^d = kY - jr$$
# 
# Se tiene previamente que en la oferta:
# 
# $$M^s = \frac{M_o^s}{p}$$
# 
# Para obtener el equilibrio:
# 
# $$M^d = M^s$$
# 
# Entonces, se igualan las ecuaciones:
# 
# $$kY - jr = \frac{M_o^s}{p}$$
# 
# $$ky - \frac{M_o^s}{p} = jr$$
# 
# $$\frac{kY}{j} - \frac{M_o^s}{pj} = r$$
# 
# $$r = - \frac{M_o^s}{pj} + \frac{kY}{j}$$
# 
# Esta última ecuación representa todos los puntos de equilibrio en el mercado monetario
# 
# Por lo tanto, la ecuación que representa a la curva LM en función de la tasa de interés que equilibra el mercado monetario es:
# 
# $$r = - \frac{1}{j} \frac{M^o_s}{P_o} + \frac{k}{j}$$

# In[9]:


######graficando:

#1----------------------Equilibrio mercado monetario

    # Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35

r = np.arange(r_size)


    # Ecuación
def Ms_MD(k, j, P, r, Y):
    Ms_MD = P*(k*Y - j*r)
    return Ms_MD

Ms_MD = Ms_MD(k, j, P, r, Y)


    # Nuevos valores de Y
Y1 = 45

def Ms_MD_Y1(k, j, P, r, Y1):
    Ms_MD = P*(k*Y1 - j*r)
    return Ms_MD

Ms_Y1 = Ms_MD_Y1(k, j, P, r, Y1)


Y2 = 25

def Ms_MD_Y2(k, j, P, r, Y2):
    Ms_MD = P*(k*Y2 - j*r)
    return Ms_MD

Ms_Y2 = Ms_MD_Y2(k, j, P, r, Y2)


# In[10]:


# Parameters
Y_size = 100

k = 0.5
j = 0.2                
P  = 10               
Ms = 30            

Y = np.arange(Y_size)


# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[11]:


# Gráfico de la derivación de la curva LM a partir del equilibrio en el mercado monetario

    # Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 8)) 


#---------------------------------
    # Gráfico 1: Equilibrio en el mercado de dinero
    
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')
ax1.plot(Y, Ms_MD, label= '$L_0$', color = '#820000')
ax1.plot(Y, Ms_Y1, label= '$L_1$', color = '#820000')
ax1.plot(Y, Ms_Y2, label= '$L_2$', color = '#820000')
ax1.axvline(x = 45,  ymin= 0, ymax= 1, color = "grey")

ax1.axhline(y=35, xmin= 0, xmax= 1, linestyle = ":", color = "black")
ax1.axhline(y=135, xmin= 0, xmax= 1, linestyle = ":", color = "black")
ax1.axhline(y=85, xmin= 0, xmax= 1, linestyle = ":", color = "black")

ax1.text(47, 139, "C", fontsize = 12, color = 'black')
ax1.text(47, 89, "B", fontsize = 12, color = 'black')
ax1.text(47, 39, "A", fontsize = 12, color = 'black')

ax1.text(0, 139, "$r_2$", fontsize = 12, color = 'black')
ax1.text(0, 89, "$r_1$", fontsize = 12, color = 'black')
ax1.text(0, 39, "$r_0$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()
 

#---------------------------------
    # Gráfico 2: Curva LM
    
ax2.set(title="LM SCHEDULE", xlabel=r'Y', ylabel=r'r')
ax2.plot(Y, i, label="LM", color = '#B9005B')

ax2.axhline(y=160, xmin= 0, xmax= 0.69, linestyle = ":", color = "black")
ax2.axhline(y=118, xmin= 0, xmax= 0.53, linestyle = ":", color = "black")
ax2.axhline(y=76, xmin= 0, xmax= 0.38, linestyle = ":", color = "black")

ax2.text(67, 164, "C", fontsize = 12, color = 'black')
ax2.text(51, 122, "B", fontsize = 12, color = 'black')
ax2.text(35, 80, "A", fontsize = 12, color = 'black')

ax2.text(0, 164, "$r_2$", fontsize = 12, color = 'black')
ax2.text(0, 122, "$r_1$", fontsize = 12, color = 'black')
ax2.text(0, 80, "$r_0$", fontsize = 12, color = 'black')

ax2.text(72.5, -14, "$Y_2$", fontsize = 12, color = 'black')
ax2.text(56, -14, "$Y_1$", fontsize = 12, color = 'black')
ax2.text(39, -14, "$Y_0$", fontsize = 12, color = 'black')

ax2.axvline(x=70,  ymin= 0, ymax= 0.69, linestyle = ":", color = "black")
ax2.axvline(x=53,  ymin= 0, ymax= 0.53, linestyle = ":", color = "black")
ax2.axvline(x=36,  ymin= 0, ymax= 0.38, linestyle = ":", color = "black")

ax2.yaxis.set_major_locator(plt.NullLocator())   
ax2.xaxis.set_major_locator(plt.NullLocator())

ax2.legend()

plt.show()


# 2. ¿Cuál es el efecto de una disminución en la Masa Monetaria $∆M_s<0$ ? Explica usando la intuición y gráficos.

# ##### Explicando:
# 
# Usando la intuición:
# Se tiene que hay un efecto de disminución en la oferta real del dinero o, en otras palabras, que se ha aplicado una política monetaria contractiva (la cantidad real de dinero disminuye).
# 
# Además, se tiene que la tasa de interés aumenta porque la oferta de dinero ($M^s$)se mueve para la izquierda y con respecto a $M^d$. En otras palabras, el punto en el que demanda y oferta se encuentren en r, se dará más arriba (esto no implica que $M^d^cambie.

# In[12]:


##### Graficando:
#--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 700             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM

# Definir SOLO el parámetro cambiado
Ms = 10

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)
# Dimensiones del gráfico
y_max = np.max(i)
v = [0, Y_size, 0, y_max]   
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, i, label="LM", color = '#E14D2A')
ax.plot(Y, i_Ms, label="LM_Ms", color = '#CD104D', linestyle = 'dashed')

# Texto agregado
plt.text(47, 85, '∆$M^s$', fontsize=12, color='#9C2C77')
plt.text(47, 75, '<-', fontsize=15, color='#9C2C77')

# Título y leyenda
ax.set(title = "Disminución en la Oferta del dinero $(M^s)$", xlabel=r'Y', ylabel=r'r')
ax.legend()


plt.show()


# 3. ¿Cuál es el efecto de un aumento en k $∆k>0$? Explica usando intuición y gráficos.

# ##### Explicando:
# 
# Usando la intuición; como la k se encuentra en la demanda:
# Teniendo en cuenta a la transacción y a la precaución:
# 
# $$L_1 = kY$$
# 
# En esta primera sección k es la elasticidad del ingreso y es importante porque en dicha ecuación -sin tomar en cuenta aún a la especulación-, k será un determinante pues, es la elasticidad del total de ingresos dentro de una economía. En otras palabras, mientras más grande sea el K, mayor cantidad de dinero se va a demandar en cuanto se incremente el nivel de ingreso de la economía.
# 
# Ejemplo:
# Si la economía funciona bien y/o mejora, surge el efecto de que las personas tienen el incentivo de obtener mayores cantidades de dinero (se demanda más dinero).  
# 
# Sin embargo, también se debe tomar en cuanta a la especulación: 
# 
# $$L_2 = -ji$$
# 
# Por lo que se tiene que:
# 
# $$M^d = L_1 + L_2$$
# 
# $$M^d = kY - ji$$
# 
# Para esto, se debe saber que M^d y K son directamente proporcionales, por lo que: si se encuentra un aumento en K, entonces $M^d$ también aumentará. 
# 
# Finalmente en el mercado monetario, si la gente demanda más dinero (porque la economía va bien), para la cantidad de dinero dada, la tasa de interés sube.

# In[13]:


#### Graficando
#--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 100             
P  = 10               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM

# Definir SOLO el parámetro cambiado
k = 12

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)

# Dimensiones del gráfico
y_max = np.max(i)
v = [0, Y_size, 0, y_max]   
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, i, label="LM", color = '#1C6758')
ax.plot(Y, i_Ms, label="LM_K", color = '#D6CDA4', linestyle = 'dashed')

# Texto agregado
plt.text(60, 400, '∆$k$', fontsize=12, color='black')
plt.text(60, 370, '<-', fontsize=15, color='grey')

# Título y leyenda
ax.set(title = "Incremento en la propensión marginal a demandar dinero △k", xlabel=r'Y', ylabel=r'r')
ax.legend()


plt.show()

