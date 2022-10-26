#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from causalgraphicalmodels import CausalGraphicalModel


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import sympy as sy


# In[4]:


import pandas as pd


# In[5]:


import math


# In[6]:


import sklearn


# In[7]:


import scipy as sp


# In[8]:


import networkx


# In[9]:


import statsmodels.api as sm


# In[10]:


import statsmodels.formula.api as smf


# In[11]:


from statsmodels.iolib.summary2 import summary_col


# REPORTE 1
# ALUMNA: CAMILA DURAND
# CODIGO: 20200918

# #EJERCICIO 1
# Matematicamente
# $$\frac{∆C}{∆Y^d}=b=0.5$$
# 
# $$∆C=C1-Co$$
# $$C1=30      Co=20$$
# $$∆C=10$$
# 
# $$∆Y^d=Y1-Yo$$
# $$Y1=40     Y=20$$
# $$∆Y^d=20$$
# 
# $$\frac{∆C}{∆Y^d}=\frac{10}{20}=b=0.5$$
# 

# Intuitivamente entendemos que:
# $$Co↑ → DA↑ → DA > Y → Y↑$$

# In[12]:


Y_size=100

b=0.5
Co=10

Yd=np.arange(Y_size)
Yd


# In[13]:


def C(Co, b, Yd):
    C = Co + b*Yd
    return C

C = C(Co, b, Yd)
C


# In[14]:


plt.plot(Yd, C, color = "red")

plt.title("Grafico 4.2- funcion de demanda de consumo")
plt.xlabel("C")
plt.ylabel("Yd")
plt.grid()

texto1 = plt.text(30,55, r'$PMgC=\frac{\Delta C}{\Delta Y}=b=0.5$', fontsize=15)
texto1 = plt.text(30,15, r'$\Delta Y^d$', fontsize=15)
texto1 = plt.text(42,25, r'$\Delta C$', fontsize=15)

texto2 = plt.text(20,19, r'$P------$', fontsize=16)
texto2 = plt.text(39,30, r'$|$', fontsize=16)
texto2 = plt.text(39,21, r'$|$', fontsize=16)
texto2 = plt.text(39,26, r'$|$', fontsize=16)

plt.show()


# EJERCICIO 2
# Intuitivamente sabemos que  
# $$∆I= Io - ∆hr$$
# $$I↓ → DA↓ → DA > Y → Y↓$$

# In[15]:


#defino los parametros

Io=100
h=0.5
R_size=50

r=np.arange(R_size)
r


# In[16]:


def I(Io, h, r):
    I = Io - h*r
    return I

I = I(Io, h, r)
I


# In[17]:


plt.plot(r, I, color = "green")

plt.title("I = Io -hr \n \n Grafico 4.3 \n Funcion de demanda de inversion")
plt.xlabel("r")
plt.ylabel("I")
plt.xticks([])
plt.yticks([])


# Ejercicio 3 
# Los supuestos del modelo ingreso-gasto keynesiano se basan principalmente en ser un modelo de corto plazo,con un nivel de precios fijo,  ademas el nivel del producto se adapta a la demanda agregada y la tasa de interés está determinada fuera del modelo, es decir, es un factor exogeno
# 

# In[18]:


#ejercicio 4
Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8 # b > m
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)
a = 2.5 

def L_45(a, Y):
    L_45 = a*Y
    return L_45

L_45 = L_45(a, Y)


# In[19]:


y_max = np.max(DA_IS_K)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, DA_IS_K, label = "DA", color = "#3D59AB") #Demanda agregada
ax.plot(Y, L_45, color = "#404040") #Línea de 45º

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Líneas punteadas punto de equilibrio
plt.axvline(x=70.5,  ymin= 0, ymax= 0.69, linestyle = ":", color = "grey")
plt.axhline(y=176, xmin= 0, xmax= 0.7, linestyle = ":", color = "grey")
plt.axhline(y=145, xmin= 0, xmax= 0.8, linestyle = ":", color = "grey")

# Texto agregado
plt.text(85, 192, '$DA=α_o + α_1Y$', fontsize = 11.5, color = 'black')
plt.text(82, 170, '$α_1=(b-m)(1-t) $', fontsize = 11.5, color = 'black')
plt.text(70, 110, '$α_o=(Co+Io+Go+Xo-hr) $', fontsize = 11.5, color = 'black')
plt.text(70, 130, '$↑$', fontsize = 18, color = 'black')
plt.text(0, 180, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(0, 152, '$α_o$', fontsize = 15, color = 'black')
    # línea 45º
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')

# Título y leyenda
ax.set(title="El ingreso de Equilibrio a Corto Plazo", xlabel= r'Y', ylabel= r'DA')
ax.legend() #mostrar leyenda

plt.show()


# Entonces entendemos que el equilibrio Y^e
# 
# $$Y^e= DA$$
# $$DA= (Co+Io+Go+Xo-hr)+[(b-m)*(1-t)]$$
# $$Y^e= \frac{1}{1-(b-m)(1-t)}*(Co+Io+Go+Xo-hr)$$
# 
# 

# Intuitivamente entendemos que para hallar el punto de equilibrio del ingreso es necesario que se acople a la demanda agregada mediante el inremento o reduccion de su produccion total

# In[20]:


# Ejercicio 5 Curva de ingreso de equilibrio ORIGINAL

    # Parámetros
Y_size = 100 

Co = 35
Io = 40
Go = 40
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

    # Ecuación 
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


#--------------------------------------------------
# NUEVA curva de ingreso de equilibrio

    # Definir SOLO el parámetro cambiado
Go = 80

# Generar la ecuación con el nuevo parámetro
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_G = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)
# 45°

a = 2.5 

def L_45(a, Y):
    L_45 = a*Y
    return L_45

L_45 = L_45(a, Y)


# In[21]:


# Gráfico
y_max = np.max(DA_IS_K)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(Y, DA_IS_K, label = "DA", color = "#3D59AB") #curva ORIGINAL
ax.plot(Y, DA_G, label = "DA_G", color = "#EE7600", linestyle = 'dashed') #NUEVA curva
ax.plot(Y, L_45, color = "#404040") #línea de 45º

# Lineas punteadas
plt.axvline(x = 75, ymin= 0, ymax = 0.75, linestyle = ":", color = "grey")
plt.axhline(y = 188, xmin= 0, xmax = 0.75, linestyle = ":", color = "grey")
plt.axvline(x = 56,  ymin= 0, ymax = 0.56, linestyle = ":", color = "grey")
plt.axhline(y = 139, xmin= 0, xmax = 0.55, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 150, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 200, '$DA_G$', fontsize = 11.5, color = '#EE7600')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(57, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(78, 0, '$Y_G$', fontsize = 12, color = '#EE7600')
plt.text(60, 45, '$→$', fontsize = 18, color = 'grey')
plt.text(20, 140, '$↑$', fontsize = 18, color = 'grey')

# Título y leyenda
ax.set(title = "Aumento del Gasto del Gobierno $(G_0)$", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# - Mathematicamente: $∆G_0 > 0  →  ¿∆Y?$
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t)} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# $$ ∆Y = \frac{1}{1 - (b - m)(1 - t)} (∆C_0 + ∆I_0 + ∆G_0 + ∆X_0 - ∆hr) $$

# como son constantes $C_0$, $I_0$, $X_0$, $h$ ni $r$, then: 
# 
# $$∆C_0 = ∆I_0 = ∆X_0 = ∆h = ∆r > 0$$
# 
# $$ ∆Y = \frac{1}{1 - (b - m)(1 - t)} (∆G_0) $$
# 
# $$ \frac{∆Y}{∆G_0}= \frac{1}{1 - (b - m)(1 - t)} $$

# Teniendo en cuenta que $∆G_0 > 0$:
# 
# $$ \frac{∆Y}{(-)}= ∆ > 0 $$
# 
# $$ ∆Y > 0 $$

# Intuitivamente:
#     $$ ↑Go → ↑DA → DA < Y → ↑Y $$

# In[22]:


# ejercicio 5 Política fiscal expansiva con una reducción de la Tasa de Tributación

    # Parámetros
Y_size = 100 

Co = 20
Io = 30
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.5 #tasa de tributación
r = 0.9

Y = np.arange(Y_size)

    # Ecuación 
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


#--------------------------------------------------
# NUEVA curva de ingreso de equilibrio

t = 0.01

# Generar la ecuación con el nuevo parámetros
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_t = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


# In[23]:


# Gráfico
y_max = np.max(DA_IS_K)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(Y, DA_IS_K, label = "DA", color = "#3D59AB") #curva ORIGINAL
ax.plot(Y, DA_t, label = "DA_t", color = "#EE7600", linestyle = 'dashed') #NUEVA curva
ax.plot(Y, L_45, color = "#404040") #línea de 45º

# Lineas punteadas
plt.axvline(x = 64, ymin= 0, ymax = 0.64, linestyle = ":", color = "grey")
plt.axhline(y = 139, xmin= 0, xmax = 0.55, linestyle = ":", color = "grey")
plt.axvline(x = 55,  ymin= 0, ymax = 0.55, linestyle = ":", color = "grey")
plt.axhline(y = 160, xmin= 0, xmax = 0.64, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 145, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 166, '$DA_t$', fontsize = 11.5, color = '#EE7600')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(56, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(66, 0, '$Y_t$', fontsize = 12, color = '#EE7600')
plt.text(60, 45, '$→$', fontsize = 18, color = 'grey')
plt.text(20, 150, '$↑$', fontsize = 18, color = 'grey')

# Título y leyenda
ax.set(title = "Reducción de la Tasa de Tributación", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# - Matematicamente: $∆t<0 → Y?$
# $$Co, Io, Go, Xo, h, r, b, m, t = symbols('Co Io Go Xo h r b m t')$$
# 
# $$f = (Co + Io + Go + Xo - h*r)/(1-(b-m)*(1-t))$$
# 
# 
# $$df_t = diff(f, t)$$
# $$df_t #∆Y/∆t$$

# $$ \frac{(-b+m)(Co+Go+Io+Xo-hr)}{(-(1-t)(b-m)+1)^2} $$
# 

# Considerando el diferencial de ∆t
# $$ \frac{(∆Y)}{∆t} = ∆$$

# Sabiendo que b > m, entonces $(m-b)<0$
# 
# Los componentes autónomos no cambian: $∆Co=∆Io=∆Xo=∆h=∆r=0$
# 
# Cualquier número elevado al cuadrado será positivo: $(1-(1-t)(b-m)+1)^2>0$
# 
# $$\frac{∆Y}{∆t}= \frac{(-)}{(+)}$$

# Dado que ∆t<0, la división de dos positivos da otro positivo:
#    $$\frac{∆Y}{(-)}= \frac{(-)}{(+)}$$
#     
#    $$∆Y= \frac{(-)(-)}{(+)}$$
#      
#    $$∆Y>0$$

# -Intuitivamente: 
# $$t↓→Co↑→DA↑→DA>Y→Y↑$$

# #### 6. - Grafique la Funcion de demanda Agregada y la recta de 45 grados señalando los valores de intercepto y pendiente como en el apartado (4).

# In[24]:


#Parametros 
Y_size = 100 

Co = 35
Io = 40
Go = 70
g = 0.2 #solo valores entre 0-0.4
Xo = 2
h = 0.7
b = 0.8 # b > m
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_C = (Co  + Io + Go + Xo - h*r) + [(b-m)*(1-t)-g]*Y
    return DA_C

DA_Cont = DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# In[25]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(DA_Cont)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(DA_Cont, label = "DA", color = "#3D59AB") #Demanda agregada
ax.plot(L_45, color = "#404040") #Línea de 45º

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Líneas punteadas punto de equilibrio
plt.axvline(x=64,  ymin= 0, ymax= 0.63, linestyle = ":", color = "grey")
plt.axhline(y=161, xmin= 0, xmax= 0.63, linestyle = ":", color = "grey")
plt.axhline(y=145, xmin= 0, xmax= 1, linestyle = ":", color = "grey")

# Texto agregado
    # punto de equilibrio
plt.text(0, 165, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(65, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(0, 135, '$α_o$', fontsize = 15, color = 'black')
    # línea 45º
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
    # ecuaciones
plt.text(82, 185, '$DA = α_0 + α_1 Y$', fontsize = 11.5, color = '#3D59AB')
plt.text(75, 151, '$α_1 = [(b-m)(1-t)-g]$', fontsize = 11.5, color = 'black')
plt.text(73, 125, '$α_0 = C_o + I_o + G_o + X_o - hr$', fontsize = 11.5, color = 'black')

plt.text(87, 175, '$↓$', fontsize = 13, color = 'black')
plt.text(85, 135, '$↑$', fontsize = 13, color = 'black')

# Título y leyenda
ax.set(title="El ingreso de Equilibrio a Corto Plazo con regla contracíclica", xlabel= r'Y', ylabel= r'DA')
ax.legend() #mostrar leyenda

plt.show()


# #### politica fiscal expansiva

# In[26]:


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
r = 0.9
g = 0.2

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_C = (Co  + Io + Go + Xo - h*r) + [(b-m)*(1-t)-g]*Y
    return DA_C

DA_Cont = DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# Nueva curva

Go = 100

def DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_C = (Co  + Io + Go + Xo - h*r) + [(b-m)*(1-t)-g]*Y
    return DA_C

DA_C_G = DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# In[27]:


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
A = [DA_Cont[0], Y[0]] # DA, coordenada inicio
B = [DA_Cont[-1], Y[-1]] # DA, coordenada fin

C = [L_45[0], Y[0]] # L_45, coordenada inicio
D = [L_45[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección

intersec = line_intersection((A, B), (C, D))
intersec # (y,x)


# In[28]:


# coordenadas de las curvas (x,y)
A = [DA_C_G[0], Y[0]] # DA, coordenada inicio
B = [DA_C_G[-1], Y[-1]] # DA, coordenada fin

C = [L_45[0], Y[0]] # L_45, coordenada inicio
D = [L_45[-1], Y[-1]] # L_45, coordenada fin

   # creación de intersección

intersec_G = line_intersection((A, B), (C, D))
intersec_G # (y,x)


# In[29]:


# Gráfico
y_max = np.max(DA_Cont)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(DA_Cont, label = "DA", color = "#3D59AB") #curva ORIGINAL
ax.plot(DA_C_G, label = "DA_G", color = "#EE7600", linestyle = 'dashed') #NUEVA curva
ax.plot(L_45, color = "#404040") #línea de 45º

# Lineas punteadas
plt.axhline(y=intersec[0], xmin= 0, xmax= 0.64, linestyle = ":", color = "grey")
plt.axvline(x=intersec[1], ymin= 0, ymax= 0.64, linestyle = ":", color = "grey")

plt.axhline(y=intersec_G[0], xmin= 0, xmax= 0.76, linestyle = ":", color = "grey")
plt.axvline(x=intersec_G[1], ymin= 0, ymax= 0.76, linestyle = ":", color = "grey")


# Texto agregado
plt.text(0, 135, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 182, '$DA_G$', fontsize = 11.5, color = '#EE7600')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(60, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(72, 0, '$Y_G$', fontsize = 12, color = '#EE7600')
plt.text(70, 45, '$→$', fontsize = 18, color = 'grey')
plt.text(20, 165, '$↑$', fontsize = 18, color = 'grey')

# Título y leyenda
ax.set(title = "Incremento del Gasto del Gobierno $(G_0)$", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# - ¿Cuál es el papel que juega el parametro en el multiplicador keynesiano?
# El parámetro  de la regla contracíclica aparece aumentando la magnitud del denominador del multiplicador keynesiano  y, por lo tanto, reduciendo su tamaño. Al reducir el tamaño de  la pendiente de la curva de Demanda Agregada  también se reducirá.
# 
# - ¿El tamaño del efecto de la politica fiscal encontrado en 6 es el mismo que en el apartado 5?
# No, el tamaño del efecto de la política fiscal del ejercicio 6) es menor al efecto del ejercicio 5). Esto se debe a que el valor del multiplicador keynesiano  se ha reducido por la adición de  en el denominador. Entonces, si  es más pequeño, el impacto de una política fiscal expansiva  será menor cuando se utilice la regla contracíclica (la adición de ). De allí que la pendiente de la curva de la Demanda Agregada  sea menor en el apartado 6).

# In[30]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
g = 0.2 #solo valores entre 0-0.4
Xo = 15
h = 0.7
b = 0.8 # b > m
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_C = (Co  + Io + Go + Xo - h*r) + [(b-m)*(1-t)-g]*Y
    return DA_C

DA_Cont = DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# Nueva curva

Xo = 1

def DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_C = (Co  + Io + Go + Xo - h*r) + [(b-m)*(1-t)-g]*Y
    return DA_C

DA_C_X = DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# In[31]:


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
A = [DA_Cont[0], Y[0]] # DA, coordenada inicio
B = [DA_Cont[-1], Y[-1]] # DA, coordenada fin

C = [L_45[0], Y[0]] # L_45, coordenada inicio
D = [L_45[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección

intersec = line_intersection((A, B), (C, D))
intersec # (y,x)


# In[32]:


# coordenadas de las curvas (x,y)
A = [DA_C_X[0], Y[0]] # DA, coordenada inicio
B = [DA_C_X[-1], Y[-1]] # DA, coordenada fin

C = [L_45[0], Y[0]] # L_45, coordenada inicio
D = [L_45[-1], Y[-1]] # L_45, coordenada fin

 # creación de intersección

intersec_X = line_intersection((A, B), (C, D))
intersec_X # (y,x)


# In[33]:


# Gráfico
y_max = np.max(DA_Cont)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(DA_Cont, label = "DA", color = "#3D59AB") #curva ORIGINAL
ax.plot(DA_C_X, label = "DA_X", color = "#EE7600", linestyle = 'dashed') #NUEVA curva
ax.plot(L_45, color = "#404040") #línea de 45º

# Lineas punteadas
plt.axhline(y=intersec[0], xmin= 0, xmax= 0.68, linestyle = ":", color = "grey")
plt.axvline(x=intersec[1], ymin= 0, ymax= 0.68, linestyle = ":", color = "grey")

plt.axhline(y=intersec_X[0], xmin= 0, xmax= 0.63, linestyle = ":", color = "grey")
plt.axvline(x=intersec_X[1], ymin= 0, ymax= 0.63, linestyle = ":", color = "grey")


# Texto agregado
plt.text(0, 180, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 135, '$DA_X$', fontsize = 11.5, color = '#EE7600')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(71, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(60, 0, '$Y_X$', fontsize = 12, color = '#EE7600')
plt.text(65.5, 45, '$←$', fontsize = 15, color = 'grey')
plt.text(20, 165, '$↓$', fontsize = 15, color = 'grey')

# Título y leyenda
ax.set(title = "Reducción de las Exportaciones", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# 
# $$REPORTE DE LAS VACAS FLACAS DE LA ECONOMIA PERUANA$$
# 
# El artículo de “las vacas flacas en la economía peruana'' explica el choque externo adverso sufrido por la economía peruana y su impacto recesivo e inflacionario durante el 2014 y 2015. Dicho texto analiza los efectos macroeconómicos que trae a nuestro país la dependencia a los metales y las políticas monetarias y fiscales que se realizaron como respuesta, además de realizar recomendaciones para enfrentar períodos recesivos para el entonces gobierno entrante de PPK y otros futuros.
# 
# Las principales fortalezas que puedo identificar del documento son su amplio análisis a través de los impactos que la economía peruana ha vivido históricamente y las respuestas que han tenido.  Su análisis no solo recae en analizar los sucedido en 2014 y 2015 sino que también lo compara con lo sucedido en desaceleraciones económicas durante años anteriores. Este análisis no sólo observa los cambios en el pbi sino también los efectos internos que eso conlleva como la disminución del empleo urbano. Dicha perspectiva ayuda a entender el enorme efecto que tiene el contexto internacional en la economía interna y como sin respuestas adecuadas las crisis sólo tenderán a expandirse. Dancourt no sólo detalla los impactos que generó el declive del precio de los metales en nuestro país sino también cómo actuaron las políticas fiscales y monetarias frente a ello, basado en una perspectiva normativa y apoyado en algunos gráficos. Y en base a todo lo revisado no se limita a hablar de presente sino que adicionalmente plantea los posibles retos que afrontaremos, remarcando los errores pasados para evitar repetirlos en el futuro cercano con el gobierno entonces venidero.
# 
# El texto analizado ayuda a entender en gran manera la dependencia a los metales que tiene nuestro país, asimismo aporta una visión amplia para el contexto en el que se escribió. Dancourt no solo explicó lo sucedido y como la caída del valor de las exportaciones generó la caída de la inversión privada y pública sino que también gracias a la data analizada ofreció una proyección a futuro cercano de la caída del 33% de la inversión minera en el 2016. Esta información ayudaba a los tomadores de decisiones y a los académicos en general, a vislumbrar cuáles serían las mejores salidas que tenía el país para responder con políticas monetarias y fiscales efectivas. Sin embargo, uno de los puntos débiles del artículo es que pudo ir más allá y examinar el contexto internacional a mayor plenitud, entendiendo el porqué de la caída del precio de los metales y como el aumento o reducción de demanda de ciertos países o grupos de países puede afectar gravemente la economía nacional. Asimismo, el artículo pudo detenerse a examinar los intentos de diversificación económica previa, para de esta manera comprender en mayor manera la profundidad de la dependencia a los metales que presenta nuestro país, basándose en examinaciones previas como la de zevallos, villarreal, del carpio y Abbara que sienta las bases para entender la crisis en el sistema financiero global.
# 
# El trabajo a seguir después del artículo es evaluar las políticas posteriores a estos años y ver si estas cambiaron y ayudaron a mitigar el impacto negativo en la economía interna del país. Asimismo, es importante seguir examinando la inversión pública pro-cíclica que exacerba los impactos tanto positivos como negativos de la economía mundial, generando tanto épocas de vacas gordas como de vacas flacas. El análisis de los sucesos es sumamente relevante, particularmente si se revisa que es lo que se puede mejorar en materia de respuestas, ya que si bien se reconoce que los impactos de la economía mundial son inevitables es importante saber cómo manejarlos de tal manera que internamente se alcance el equilibrio. Asimismo, es relevante examinar los intentos de dinamizar y diversificar la economía y como hasta la fecha aún no se tienen resultados positivos, como lo menciona Hunt, lo cual nos enfrenta a una gran dependencia y expandir dichos estudios a la región latinoamericana para tener de referentes economías con contextos similares y como sus diversas repuestas pudieron ayudar a mitigar el choque adverso o no.
# 
# 
# Hunt, S. J. (2020). La formación de la economía peruana, distribución y crecimiento en la historia de peru y america latina (1.a ed.). IEP.
# 
# Zevallos, M., Villarreal, F., del Carpio, C., & Abbara, O. (2014). Influencia de los Precios de los Metales y el Mercado Internacional en el Riesgo Bursátil Peruano. BANCO CENTRAL DE RESERVA DEL PERÚ.
# 

# In[ ]:




