import numpy as np 
import random as rand
import math
from pylab import *

Lx = 12000000
Ly = 6000000
delta_s = 200000
omega = 7.2921*10**(-5)
f_0 = 2 * omega * math.sin(math.pi/4)
k = (2 * math.pi)/(Lx/2)
j = (2 * math.pi)/(Ly/2)
g = 9.81

xvalues, yvalues = np.meshgrid(np.arange(0,Lx,delta_s),np.arange(0,Ly,delta_s)) #Crée deux tableaux de 30 pour 60, l'un avec les valeurs de x et l'autre de y

psi_0 = np.zeros((len(xvalues),len(xvalues[0])))
zeta_0 = np.zeros((len(xvalues),len(xvalues[0])))
u = np.zeros((len(xvalues),len(xvalues[0])))
v = np.zeros((len(xvalues),len(xvalues[0])))

############ - CI - ############

for i in range(len(xvalues)):
	for l in range(len(xvalues[i])):
		psi_0[i,l] = (g/f_0) * (100 * math.sin(k * xvalues[i,l]) * math.cos(j * yvalues[i,l]))

		# On évalue zeta en utilisant la formule (4) et en imposant la périodicité aux bords.

		if i != len(psi_0) - 1 and l != len(psi_0[i]) - 1:
			zeta_0[i,l] = (1/(delta_s**2)) * (-4 * psi_0[i,l] + psi_0[i+1 , l] + psi_0[i-1 , l] + psi_0[i , l+1] + psi_0[i , l-1]) 
		if i == len(psi_0) - 1:
			zeta_0[i,l] = zeta_0[0,l]
		if l == len(psi_0[i]) - 1:
			zeta_0[i,l] = zeta_0[i,0]

		# On évalue enfin les champs de vitesse x et y toujours avec périodicité aux bords.

		if i != len(psi_0) - 1 and l != len(psi_0[i]) - 1:
			u[i,l] = -(psi_0[i+1,l] - psi_0	[i-1,l])/(2*delta_s)
			v[i,l] = (psi_0[i,l+1] - psi_0[i,l-1])/(2*delta_s)
		if i == len(psi_0) - 1:
			u[i,l] = u[0,l]
			v[i,l] = v[0,l]
		if l == len(psi_0[i]) - 1:
			u[i,l] = u[i,0]
			v[i,l] = v[i,0]

pcolormesh(xvalues,yvalues,zeta_0)
colorbar()
title('zeta')
show()
pcolormesh(xvalues,yvalues,psi_0)
colorbar()
title('streamfunction')
show()
quiver(xvalues,yvalues,u,v)
colorbar()
show()
