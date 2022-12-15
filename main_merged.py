import numpy as np
import random as rand
import math
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

###################################################### Paramètres de la simulation ##################################################
phi_0 = ((2*np.pi)/360)*45  # latitude en radian
Lx = 12000000
Ly = 6000000
Delta_s = 200000  # résolution de la maille spatiale en mètre. Valeur par défaut = 200km
Delta_t = 3600    # résolution de la maille temporelle en seconde valeur par défaut d'une heure.
nbr_jours = 1  # nombre de jours de simulation
T = 86400*nbr_jours  # temps total de simulation en seconde
M = int(Lx/Delta_s)  # nombre d'itération selon x
N = int(Ly/Delta_s)  # nombre d'itération selon y
K = int(T/Delta_t)  # nombre d'itération selon t
Wx = 6000000  # longueurs d'onde champ initial selon x en mètre
Wy = 3000000  # longueurs d'onde champ initial selon y en mètre
Omega = 7.2921*10**(-5)  # vitesse angulaire de la rotation de la Terre
g = 9.81  # norme de l'accélaration gravitationnelle
a = 6371000  # rayon moyen de la Terre en mètres

###################################################### Discrétisation des mailles ##################################################
# Crée deux tableaux de taille NxM, l'un avec les valeurs discrétisée de x et l'autre de y
xvalues, yvalues = np.meshgrid(np.arange(0, Lx, Delta_s), np.arange(0, Ly, Delta_s))

###################################################### Fonctions de la simulation ##################################################
"""
Note sur la périodicité:
	Les modules (%) sont utilisés pour faire la périodicité aux bords. Ainsi pour un psi situé par exemple sur le bord est il prendra en compte pour son calcul
son voisin fictif situé sur le bord ouest.
"""


def f_0_scal(phi_0):
    """Donne la valeur du paramètre de Coriolis f_0 en un scalaire à phi_0 fixé"""
    return 2*Omega*np.sin(phi_0)


def beta_scal(phi):
    """Donne la valeur du paramètre beta pour une valeur de latitude phi donnée"""
    return (2*Omega*np.cos(phi)/a)


def psi_init():
    """Donne une condition initiale pour la fonction psi_0"""
    k = (2*np.pi)/Wx
    j = (2*np.pi)/Wy
    x = np.arange(0, Lx, Delta_s)  # discrétisation de l'axe horizontal
    y = np.arange(0, Ly, Delta_s)  # discrétisation de l'axe vertical
    x_grid = np.matrix(x,)
    y_grid = np.matrix(y,)
    psi_0_T = np.zeros((N, M))
    psi_0_T = (g/f_0_scal(phi_0))*(100*np.sin(k*(x_grid.T))@(np.cos(j*y_grid)))
    psi_0 = psi_0_T.T
    return psi_0


def zeta_init(psi):
    """Donne la composante verticale de la vorticité relative en fonction de la fonction de courant initiale à l'aide de l'équation(2)"""
    zeta = np.zeros((N, M))
    for i in range(N):
        for l in range(M):
            zeta[i, l] = (1/(Delta_s**2)) * (-4 * psi[i, l] + psi[(i+1) %
                                                                  N, l] + psi[i-1, l] + psi[i, (l+1) % M] + psi[i, l-1])
    return zeta


def u(psi):
    """Donne la composante zonale du champ de vitesse à partir de la fonction de courant"""
    u = np.zeros((N, M))
    for i in range(N):
        for l in range(M):
            u[i, l] = (-1/(2*Delta_s))*(psi[(i+1) % N, l] - psi[i-1, l])
    return u


def v(psi):
    """Donne la composante méridienne du champ de vitesse à partir de la fonction de courant"""
    v = np.zeros((N, M))
    for i in range(N):
        for l in range(M):
            v[i, l] = (1/(2*Delta_s))*(psi[i, (l+1) % M] - psi[i, l-1])
    return v


def vort_flux(zeta, u, v):
    """Donne la valeur du flux de la composante verticale de la vorticité relative F. Prend comme argument le champ de vitesse.
Prend comme argument la matrice de la composante selon x du champ de vitesse, puis la matrice de la composante y du champs de vitesse
, la matrice de la composante verticale de la vorticité relative."""
    F = np.zeros((N, M))
    for i in range(N):
        for l in range(M):
            F[i, l] = (1/(2*Delta_s))*(zeta[i, (l+1) % M]*u[i, (l+1) % M] - zeta[i, l-1] * u[i, l-1] +
                                       zeta[(i+1) % N, l]*v[(i+1) % N, l] - zeta[i-1, l]*v[i-1, l]) + beta_scal(phi_0) * v[i, l]
    return F


def zeta(F, zeta):
    """Donne la valeur de la vorticité à un instant t à partir à du flux de vorticité et la valeur de la vorticité au temps t-delta_t.
    Sera utile dans l'intégration temporelle"""
    new_zeta = np.zeros((N, M))

    for i in range(N):
        for l in range(M):
            new_zeta[i, l] = -F[i, l] * 2 * Delta_t + zeta[i, l]
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
    A[i, (i+1) % (N*M)] = 1
    A[i, i-1] = 1
    A[i, (i+M) % (N*M)] = 1
    A[i, i-M] = 1
A_inv = np.linalg.inv(A)


def psi(zeta):
    """Fonction qui permet de calculer la fonction de courant psi à partir de la vorticité relative zeta en inversant l'opérateur du laplacien.
On transforme par différence finie l'opération du laplacien en une matrice A qui donne un ensemble de NxM équations algébriques à NxM inconnes qu'on résoud
en inversant A."""
    # On met la matrice zeta en forme de colonne en parcourant de gauche à droite et de bas en haut.
    zeta_col = np.zeros((N*M, 1))
    for i in range(N):
        for l in range(M):
            zeta_col[l+i*M, 0] = zeta[i, l]

    """Ici intervient notre matrice A définie plus haut.
	En définissant psi_col par rapport au tableau psi de la même manière que l'on a définit zeta_col par rapprot à zeta
	notre système est donc donné par A psi_col = delta_s^2 zeta_col et sa solution est trouvée en inversant la matrice A
	psi_col = A_inv delta_s^2 zeta_col.
	"""
    psi_col = Delta_s**2 * np.dot(A_inv, zeta_col)

    # Pour finir il faut remettre psi en forme de tableau pour que ce soit cohérent avec le reste de l'implémentation.
    psi = np.zeros((N, M))
    for i in range(N):
        for l in range(M):
            psi[i, l] = psi_col[l+i*M, 0]
    return psi


################################################### - Conditions Initiales - ###############################################
psi_0 = psi_init()
zeta_0 = zeta_init(psi_0)
u_0 = u(psi_0)
v_0 = v(psi_0)
# Output de la résolution des mailles
print("-----------------------------------------------------------------------------------------------")
print("Résolution numérique avec une maille spatiale de {}x{} points".format(
    M, N))
print("Résolution numérique avec une maille temporelle de {} points".format(K))
print("-----------------------------------------------------------------------------------------------")
############################################### - Affichage Conditions Initiales ############################################
########## Contourplot de la fonction de courant initiale ############
print(np.shape(psi_0))
plt.contourf(xvalues, yvalues, psi_0, 100)
plt.colorbar()
plt.title("Contour plot de la fonction de courant initiale $\psi_0(x,y)$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
    int(Lx/1000), int(Ly/1000), int(Delta_s/1000), int(Wx/1000), int(Wy/1000)), fontsize=11)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
# plt.show()
#### Contourplot de la composante verticale de la vorticité relative initiale ###
plt.contourf(xvalues, yvalues, zeta_0, 100)
plt.colorbar()
plt.title("Contour plot de $\zeta_0(x,y)$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
    int(Lx/1000), int(Ly/1000), int(Delta_s/1000), int(Wx/1000), int(Wy/1000)), fontsize=11)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
# plt.show()
######################## Plot du champ de vitesse initial ######################
slice_interval = 1  # Slicer index for smoother quiver function
skip = (slice(None, None, slice_interval), slice(None, None, slice_interval))
vels = np.hypot(u_0, v_0)  # Velocity norm of each velocity vector
Quiver = plt.quiver(xvalues[skip], yvalues[skip], u_0[skip], v_0[skip], vels[skip],
                    units='height', angles='xy')
plt.colorbar(Quiver)
# plt.quiverkey(Quiver, 1.01, 1.01, 30000, label="?m/s",
# labelcolor = 'blue', labelpos = 'N', coordinates = "axes")
plt.title("Champ de vitesse initial $(u_0(x,y),v_0(x,y))$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
    int(Lx/1000), int(Ly/1000), int(Delta_s/1000), int(Wx/1000), int(Wy/1000)), fontsize=11)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
# plt.show()
################################################## - Intégration Numérique - ###############################################
# On crée un vecteur par variable auquel on va rajouter une composante à chaque pas de temps.
nb_plot = 1
for t in range(K):
    if t == 0:  # On pose les CI
        u_tot = [u_0]
        v_tot = [v_0]
        F_tot = [vort_flux(zeta_0, u_0, v_0)]
        zeta_tot = [zeta_0]
        zeta_tot.append(np.zeros((N, M)))
        for i in range(N):
            for l in range(M):
                zeta_tot[-1][i, l] = - F_tot[-1][i, l] * Delta_t + zeta_0[i, l]
        psi_tot = [psi(zeta_tot[-1])]
    else:
        u_tot.append(u(psi_tot[-1]))
        v_tot.append(v(psi_tot[-1]))
        F_tot.append(vort_flux(zeta_tot[-1], u_tot[-1], v_tot[-1]))
        zeta_tot.append(zeta(F_tot[-1], zeta_tot[-2]))
        psi_tot.append(psi(zeta_tot[-1]))
        print(np.shape(psi_tot))
    print("Itérations = ", t, "/", K)
################################################ Affichage Intégration Numérique #############################################
############ Subplot des intégrations temporelles version non animée et pas quali #########
for t in range(K):
    fig = plt.figure(figsize=[16/1.3, 9/1.3])
    ax_stream_func = plt.subplot2grid((2, 2), (0, 0))
    ax_U = plt.subplot2grid((2, 2), (0, 1))
    ax_V = plt.subplot2grid((2, 2), (1, 1))
    ax_vort = plt.subplot2grid((2, 2), (1, 0))
    print("itérations = ", t, "/", K)
    nbr_heures = t
    nbr_jours = nbr_heures/24
    plt.suptitle("Temps : t = {:.2f} heures = {:.2f} jours".format(nbr_heures, nbr_jours))
    ax_stream_func.contourf(xvalues, yvalues, psi_tot[-1], 100)
    # fig.colorbar(pcm)
    ax_stream_func.set_title("$\psi(x,y,t)$")
    ax_U.contourf(X, Y, U[:, :, t], 100)
    # fig[0, 1].colorbar()
    ax_U.set_title("$U(x,y,t)$")
    ax_V.contourf(X, Y, V[:, :, t], 100)
    # axs[1, 1].colorbar()
    ax_V.set_title("$V(x,y,t)$")
    ax_vort.contourf(X, Y, vort_dyn[:, :, t], 100)
    # axs[1, 0].colorbar()
    ax_vort.set_title("$\zeta(x,y,t)$")
    plt.show()


"""
######################## Début test animation cfr vidéo youtube ####################
fig = plt.figure(figsize=[16/1.3, 9/1.3])
ax_stream_func = plt.subplot2grid((2, 2), (0, 0))
ax_U = plt.subplot2grid((2, 2), (0, 1))
ax_V = plt.subplot2grid((2, 2), (1, 1))
ax_vort = plt.subplot2grid((2, 2), (1, 0))

# fig, ax_stream_func = plt.subplot()


def animate(iter):
    ax_stream_func.clear()
    ax_stream_func.contourf(X, Y, stream_func_dyn[:, :, iter])
    return ax_stream_func


ani = animation.FuncAnimation(fig, animate, frames=500, blit=True, interval=1000/24, repeat=True)

plt.show()

############# Première version manuelle affichage ################
for t in range(K):
    print("itérations = ", t)
    plt.pause(0.01)
    nbr_heures = (t*Delta_t)/60
    plt.suptitle("Temps : t = {:.2f}h".format(nbr_heures))
    axs[0, 0].contourf(X, Y, stream_func_dyn[:, :, t])
    # fig.colorbar(pcm)
    axs[0, 0].set_title("$\psi(x,y,t)$")
    axs[0, 1].contourf(X, Y, U[:, :, t])
    # fig[0, 1].colorbar()
    axs[0, 1].set_title("$U(x,y,t)$")
    axs[1, 1].contourf(X, Y, V[:, :, t])
    # axs[1, 1].colorbar()
    axs[1, 1].set_title("$V(x,y,t)$")
    axs[1, 0].contourf(X, Y, vort_dyn[:, :, t])
    # axs[1, 0].colorbar()
    axs[1, 0].set_title("$\zeta(x,y,t)$")

    for ax in axs.flat:
        ax.set(xlabel='$x$', ylabel='$y$')
    for ax in axs.flat:  # Cache label x et y pour les plots sur les côtés.
        ax.label_outer()

    plt.show()
"""


##############

# On plot les résultats tous les n pas de temps
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
	# Plot de 6 tableau le long de la simulation
	if t in np.arange(0,60,10):
		subplot(2,3,nb_plot)
		title('t= {} heures'.format(t*delta_t_en_heure))
		pcolormesh(xvalues,yvalues,zeta_tot[-1])
		colorbar()
		nb_plot += 1
"""
"""
# plot dynamique des vecteurs vitesses au cours du temps
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


# plot dynamique de psi au cours du temps, pour avoir zeta il faut changer les deux psi_tot en zeta_tot
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
"""
