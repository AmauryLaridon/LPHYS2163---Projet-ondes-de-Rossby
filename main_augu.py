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

########## - Définition des fonctions - ###########


def current_flux(zeta, u, v):
	beta = 2*omega*math.cos(math.pi/4)/earth_radius
	current_flux = np.zeros((N,M))
	for i in range(N):
		for l in range(M):
			if i != N - 1 and l != M - 1:
				current_flux[i,l] = (1/(2*delta_s))*(zeta[i,l+1]*u[i,l+1] - zeta[i,l-1]*u[i,l-1] + zeta[i+1,l]*v[i+1,l] - zeta[i-1,l]*v[i-1,l]) + beta * v[i,l]
			
			# Périodicité
			if i == N - 1:
				current_flux[i,l] = u[0,l]
			if l == M - 1:
				current_flux[i,l] = u[i,0]
			if i == N - 1 and l == M - 1:
				current_flux[i,l] = current_flux[0,0]
	return current_flux

def u(psi):
	u = np.zeros((N,M))
	for i in range(N):
		for l in range(M):
			if i != N - 1 and l != M - 1:
				u[i,l] = (-1/(2*delta_s))*(psi[i+1,l] - psi[i-1,l])

			# Périodicité
			if i == N - 1:
				u[i,l] = u[0,l]
			if l == M - 1:
				u[i,l] = u[i,0]
			if i == N - 1 and l == M - 1:
				u[i,l] = u[0,0]
	return u

def v(psi):
	v = np.zeros((N,M))
	for i in range(N):
		for l in range(M):
			if i != N - 1 and l != M - 1:
				v[i,l] = (1/(2*delta_s))*(psi[i,l+1] - psi[i,l-1])

			# Périodicité
			if i == N - 1:
				v[i,l] = v[0,l]
			if l == M - 1:
				v[i,l] = v[i,0]
			if i == N - 1 and l == M - 1:
				v[i,l] = v[0,0]
	return v

def zeta_init(psi):
	zeta = np.zeros((N,M))
	for i in range(N):
		for l in range(M):			
			if i != N - 1 and l != M - 1:
				zeta[i,l] = (1/(delta_s**2)) * (-4 * psi[i,l] + psi[i+1 , l] + psi[i-1 , l] + psi[i , l+1] + psi[i , l-1]) 

			# Périodicité
			if i == N - 1:
				zeta[i,l] = zeta[0,l]
			if l == M - 1:
				zeta[i,l] = zeta[i,0]
			if i == N - 1 and l == M - 1:
				zeta[i,l] = zeta[0,0]
	return zeta

def zeta(F,zeta):
	new_zeta = np.zeros((N,M))
	
	for i in range(N):
		for l in range(M):
			new_zeta[i,l] = -F[i,l] * 2 * delta_t + zeta[i,l]
	return new_zeta


"""
	Ici on définit une grande matrice A qui est utile pour la fonction psi(zeta) utilisée juste après. On la définit ici car 
	la fonction sera utilisée de nombreuse fois et recrée cette matrice et l'inverser à chaque fois prendrait beaucoup de temps
	de calcul pour rien. Notez que les commentaires ajoutés au cours de la création de la matrice sont à comprendre 
	dans le contexte de la fonction psi(zeta) ci-dessous
"""
# Définition d'une grande matrice carrée diagonale A de côté N x M avec -4 sur sa diagonale. 
A = -4*np.eye(N*M)
"""
On veut sur chaque ligne ajouter un terme "1" correspondant à l'emplacement des 4 valeurs adjacentes de psi_i,j. On a donc
	- Deux 1 à gauche et à droite de la diagonale, respectivement pour psi_i,j+1 et psi_i,j-1
	- Deux 1 à une distance M de la diagonale pour psi_i+1,j et psi_i-1,j
"""
for i in range(N*M):
	A[i,(i+1)%(N*M)] = 1
	A[i,i-1] = 1
	A[i,(i+M)%(N*M)] = 1		
	A[i,i-M] = 1
A_inv = np.linalg.inv(A)


def psi(zeta):
	# On met la matrice zeta en forme de colonne en parcourant de gauche à droite et de bas en haut.
	zeta_col = np.zeros((N*M,1)) 
	for i in range(N):
		for l in range(M):
			zeta_col[l+i*M,0] = zeta[i,l]

	# On définit une matrice colonne psi de taille égale
	psi_col = np.zeros((N*M,1))
	# Ici intervient notre matrice A.
	"""
	Notre système est donc donné par A psi_col = delta_s zeta_col et sa solution est trouvée en inversant la matrice A
	psi = A^-1 delta_s zeta_col.
	"""
	psi_col = delta_s**2 * np.dot(A_inv,zeta_col)

	# Pour finir il faut remettre psi en forme de tableau pour que ce soit cohérent avec le reste de l'implémentation.
	psi = np.zeros((N,M))
	for i in range(N):
		for l in range(M):
			psi[i,l] = psi_col[l+i*M,0]

	return psi


##################### - CI - ######################
psi_0 = np.zeros((N,M))
for i in range(N):
	for l in range(M):
		psi_0[i,l] = (g/f_0) * (100 * math.sin(k * xvalues[i,l]) * math.cos(j * yvalues[i,l]))

zeta_0 = zeta_init(psi_0)
u_0 = u(psi_0)
v_0 = v(psi_0)
"""
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
"""

############ - Numerical integration - ############

temps_integration_en_jour = 4
delta_t_en_heure = 1
delta_t = 3600 * delta_t_en_heure #passage au SI
temps_integration = 60 * 60 * 24 * temps_integration_en_jour #passage au SI
nb_pas_de_temps = int(temps_integration/delta_t)

#à chaque itération du temps le résultat sera ajouté en dernière place de ces vecteurs.


for t in range(nb_pas_de_temps):
	if t == 0: #On pose les CI
		u_tot = [u_0]
		v_tot = [v_0]

		F_tot = [current_flux(zeta_0,u_0,v_0)]
		zeta_tot = [zeta_0]

		zeta_tot.append(np.zeros((N,M)))
		for i in range(N):
			for l in range(M):
				zeta_tot[-1][i,l] = - F_tot[-1][i,l] * delta_t + zeta_tot[-2][i,l]

		psi_tot = [psi_0]

	else:
		u_tot.append(u(psi_tot[-1]))
		v_tot.append(v(psi_tot[-1]))
		F_tot.append(current_flux(zeta_tot[-1],u_tot[-1],v_tot[-1]))
		zeta_tot.append(zeta(F_tot[-1],zeta_tot[-2]))
		psi_tot.append(psi(zeta_tot[-1]))
	print(t,"/", nb_pas_de_temps)


	#On plot les résultats tous les n pas de temps
	n = 5
	if t in np.arange(0,nb_pas_de_temps,n) :
		pcolormesh(xvalues,yvalues,zeta_tot[-1])
		colorbar()
		title('zeta')
		show()
		pcolormesh(xvalues,yvalues,psi_tot[-1])
		colorbar()
		title('streamfunction')
		show()
		quiver(xvalues,yvalues,u_tot[-1],v_tot[-1])
		colorbar()
		show()


#################### - Plot - #####################


