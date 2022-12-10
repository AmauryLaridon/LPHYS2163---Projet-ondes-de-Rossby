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
earth_radius = 6371000000 #rayon de la terre en m
M = int(Lx/delta_s)
N = int(Ly/delta_s)

xvalues, yvalues = np.meshgrid(np.arange(0,Lx,delta_s),np.arange(0,Ly,delta_s)) #Crée deux tableaux de 30 pour 60, l'un avec les valeurs de x et l'autre de y

psi_0 = np.zeros((N,M))
zeta_0 = np.zeros((N,M))
u_0 = np.zeros((N,M))
v_0 = np.zeros((N,M))

def current_flux(zeta, u, v):
	beta = 2*omega*math.cos(math.pi/4)/earth_radius
	current_flux = np.zeros((N,M))
	for i in range(N):
		for l in range(M):
			current_flux[i,l] = (1/2*delta_s)*(zeta[i,l+1]*u[i,l+1] - zeta[i,l-1]*u[i,l-1] + zeta[i+1,l]*v[i+1,l] - zeta[i-1,l]*v[i-1,l]) + beta * v[i,l]
	return current_flux

def u(psi):
	u = np.zeros((N,M))

	if i != N - 1 and l != M - 1:
		u[i,l] = (-1/(2*delta_s))*(psi[i+1,l] - psi[i-1,l])
	if i == N - 1:
		u[i,l] = u[0,l]
	if l == M - 1:
		u[i,l] = u[i,0]
	return u

def v(psi):
	v = np.zeros((N,M))

	if i != N - 1 and l != M - 1:
		v[i,l] = (1/(2*delta_s))*(psi[i,l+1] - psi[i,l-1])
	if i == N - 1:
		v[i,l] = v[0,l]
	if l == M - 1:
		v[i,l] = v[i,0]
	return v

def zeta(psi):
	zeta = np.zeros((N,M))

	if i != N - 1 and l != M - 1:
		zeta[i,l] = (1/(delta_s**2)) * (-4 * psi[i,l] + psi[i+1 , l] + psi[i-1 , l] + psi[i , l+1] + psi[i , l-1]) 
	if i == N - 1:
		zeta[i,l] = zeta[0,l]
	if l == M - 1:
		zeta[i,l] = zeta[i,0]
	return zeta

##################### - CI - ######################

for i in range(N):
	for l in range(M):
		psi_0[i,l] = (g/f_0) * (100 * math.sin(k * xvalues[i,l]) * math.cos(j * yvalues[i,l]))

		# On évalue zeta en utilisant la formule (4) et en imposant la périodicité aux bords.

		if i != N - 1 and l != M - 1:
			zeta_0[i,l] = (1/(delta_s**2)) * (-4 * psi_0[i,l] + psi_0[i+1 , l] + psi_0[i-1 , l] + psi_0[i , l+1] + psi_0[i , l-1]) 
		if i == N - 1:
			zeta_0[i,l] = zeta_0[0,l]
		if l == M - 1:
			zeta_0[i,l] = zeta_0[i,0]

		# On évalue enfin les champs de vitesse x et y toujours avec périodicité aux bords.

		if i != N - 1 and l != M - 1:
			u_0[i,l] = (-1/(2*delta_s))*(psi_0[i+1,l] - psi_0[i-1,l])
			v_0[i,l] = (1/(2*delta_s))*(psi_0[i,l+1] - psi_0[i,l-1])
		if i == N - 1:
			u_0[i,l] = u_0[0,l]
			v_0[i,l] = v_0[0,l]
		if l == M - 1:
			u_0[i,l] = u_0[i,0]
			v_0[i,l] = v_0[i,0]

############ - Numerical integration - ############

nb_jour = 4
nb_heure = 1
delta_t = 3600 * nb_heure #passage au SI
temps_integration = 60 * 60 * 24 * nb_jour #passage au SI

#à chaque itératon du temps le résultat sera ajouté en dernière place de ces vecteurs.
u_tot = [u]
v_tot = [v]
psi_tot = [psi_0]
zeta_tot = [zeta_0]



#################### - Plot - #####################

pcolormesh(xvalues,yvalues,zeta_0)
colorbar()
title('zeta')
show()
pcolormesh(xvalues,yvalues,psi_0)
colorbar()
title('streamfunction')
show()
quiver(xvalues,yvalues,u_0,v_0)
colorbar()
show()
