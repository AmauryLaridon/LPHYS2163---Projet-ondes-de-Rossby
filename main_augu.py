import numpy as np 
import random as rand
import math
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


Lx = 12000000
Ly = 6000000
delta_s = 100000
omega = 7.2921*10**(-5)
f_0 = 2 * omega * math.sin(math.pi/4)
k = (2 * math.pi)/(Lx/2)
j = (2 * math.pi)/(Ly/2)
g = 9.81
rayon_terre = 6371000000 
beta = 2*omega*math.cos(math.pi/4)/rayon_terre
M = int(Lx/delta_s)
N = int(Ly/delta_s)

xvalues, yvalues = np.meshgrid(np.arange(0,Lx,delta_s),np.arange(0,Ly,delta_s)) #Crée deux tableaux de 30 pour 60, l'un avec les valeurs de x et l'autre de y

########## - Définition des fonctions - ###########

"""
Note sur la périodicité:
	Les modules (%) sont utilisés pour faire la périodicité aux bords. Ainsi pour un psi situé par exemple sur le bord est il prendra en compte pour son calcul
son voisin fictif situé sur le bord ouest.
"""

def current_flux(zeta, u, v):

	current_flux = np.zeros((N,M))
	for i in range(N):
		for l in range(M):
			current_flux[i,l] = (1/(2*delta_s))*(zeta[i,(l+1)%M]*u[i,(l+1)%M] - zeta[i,l-1]*u[i,l-1] + zeta[(i+1)%N,l]*v[(i+1)%N,l] - zeta[i-1,l]*v[i-1,l]) + beta * v[i,l]
			
	return current_flux

def u(psi):
	u = np.zeros((N,M))
	for i in range(N):
		for l in range(M):
			u[i,l] = (-1/(2*delta_s))*(psi[(i+1)%N,l] - psi[i-1,l])
	return u

def v(psi):
	v = np.zeros((N,M))
	for i in range(N):
		for l in range(M):
			v[i,l] = (1/(2*delta_s))*(psi[i,(l+1)%M] - psi[i,l-1])

	return v

def zeta_init(psi):
	zeta = np.zeros((N,M))
	for i in range(N):
		for l in range(M):
			zeta[i,l] = (1/(delta_s**2)) * (-4 * psi[i,l] + psi[(i+1)%N , l] + psi[i-1 , l] + psi[i , (l+1)%M] + psi[i , l-1]) 

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

	 
	"""
	Ici intervient notre matrice A définie plus haut.

	En définissant psi_col par rapport au tableau psi de la même manière que l'on a définit zeta_col par rapprot à zeta
	notre système est donc donné par A psi_col = delta_s^2 zeta_col et sa solution est trouvée en inversant la matrice A
	psi_col = A_inv delta_s^2 zeta_col.
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
subplot(1,3,1)
pcolormesh(xvalues,yvalues,zeta_0)
colorbar()
title('zeta_0')

subplot(1,3,2)
title('psi_0')
pcolormesh(xvalues,yvalues,psi_0)
colorbar()

subplot(1,3,3)
quiver(xvalues,yvalues,u_0,v_0)
colorbar()
show()
"""

############ - Numerical integration - ############

temps_integration_en_jour = 12
delta_t_en_heure = 1
delta_t = 3600 * delta_t_en_heure #passage au SI
temps_integration = 60 * 60 * 24 * temps_integration_en_jour #passage au SI
nb_pas_de_temps = int(temps_integration/delta_t)

# à chaque itération du temps le résultat sera ajouté en dernière place de ces vecteurs.

nb_plot = 1
for t in range(nb_pas_de_temps):
	if t == 0: #On pose les CI
		u_tot = [u_0]
		v_tot = [v_0]

		F_tot = [current_flux(zeta_0,u_0,v_0)]
		zeta_tot = [zeta_0]

		zeta_tot.append(np.zeros((N,M)))
		for i in range(N):
			for l in range(M):
				zeta_tot[-1][i,l] = - F_tot[-1][i,l] * delta_t + zeta_0[i,l]

		psi_tot = [psi(zeta_tot[-1])]

	else:
		u_tot.append(u(psi_tot[-1]))
		v_tot.append(v(psi_tot[-1]))
		F_tot.append(current_flux(zeta_tot[-1],u_tot[-1],v_tot[-1]))
		zeta_tot.append(zeta(F_tot[-1],zeta_tot[-2]))
		psi_tot.append(psi(zeta_tot[-1]))
	print(t,"/", nb_pas_de_temps)


	#On plot les résultats tous les n pas de temps
"""
	n = 5
	if t in np.arange(0,nb_pas_de_temps,n):
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
	#Plot de 6 tableau le long de la simulation
	if t in np.arange(0,60,10):
		subplot(2,3,nb_plot)
		title('t= {} heures'.format(t*delta_t_en_heure))
		pcolormesh(xvalues,yvalues,zeta_tot[-1])
		colorbar()

		nb_plot += 1
"""
"""
#plot dynamique des vecteurs vitesses au cours du temps

u_anim = u_tot[0]
v_anim = v_tot[0]

fig, ax = plt.subplots(1,1)
Q = ax.quiver(xvalues, yvalues, u_anim, v_anim, pivot='mid', color='r')

def update_quiver(num, Q, x, y):
	if num == nb_pas_de_temps:
		plt.pause(100)
	u_anim = u_tot[num]
	v_anim = v_tot[num]
	title('t = {} heures'.format(num*delta_t_en_heure))

	Q.set_UVC(u_anim,v_anim)

	return Q,

# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q,xvalues,yvalues),interval=50, blit=False)
fig.tight_layout()
plt.show()
"""

#plot dynamique de psi au cours du temps, pour avoir zeta il faut changer les deux psi_tot en zeta_tot
fig = plt.figure()
im = plt.imshow(psi_tot[0], interpolation='nearest', cmap='Blues')
colorbar()
def update(data):
	im.set_array(data)
def data_gen(n):
	for n in range(n):
		title('psi, temps ={}h'.format(n*delta_t_en_heure))
		yield psi_tot[n+1]
ani = animation.FuncAnimation(fig, update, data_gen(nb_pas_de_temps), interval=0)
plt.show()


