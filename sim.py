import matplotlib.pyplot as plt
import numpy as np


###################################################### Paramètres de la simulation ##################################################

phi_0 = ((2*np.pi)/360)*45  # latitude en radian
Delta_s = 200  # résolution de la maille spatiale
Delta_t = 1  # résolution de la maille temporelle
T = 360  # temps total de simulation
L_x = 12000  # longueur du domaine selon x
L_y = 6000  # longueur du domaine selon y en km
M = int(L_x/Delta_s)  # nombre d'itération selon x
N = int(L_y/Delta_s)  # nombre d'itération selon x
W_x = 6000  # longueurs d'onde champ initial selon x
W_y = 3000  # longueurs d'onde champ initial selon y
Omega = 7.292115*10**(-5)  # vitesse angulaire de la rotation de la Terre
g = 9.81  # norme de l'accélaration gravitationnelle

###################################################### Discrétisation des mailles ##################################################

x = np.arange(0, L_x, Delta_s)  # discrétisation de l'axe horizontal
y = np.arange(0, L_y, Delta_s)  # discrétisation de l'axe vertical
x_grid = np.matrix(x,)
y_grid = np.matrix(y,)
t_grid = np.arange(0, T, Delta_t)

###################################################### Fonctions de la simulation ##################################################


# Fonctions de la simulation


def f_0(Omega, phi_0):
    """Donne la valeur du paramètre de Coriolis f_0"""
    return (2*Omega*np.sin(phi_0))


def psi_init():
    """Donne une condition initiale pour la fonction psi_0"""
    k = (2*np.pi)/W_x
    j = (2*np.pi)/W_y
    psi_0_T = np.zeros((N, M))
    psi_0_T = (g/f_0(Omega, phi_0))*(100*np.sin(k*(x_grid.T))@(np.cos(j*y_grid)))
    psi_0 = psi_0_T.T
    return psi_0


def vort_init():
    """Donne la composante verticale de la vorticité relative en fonction de la fonction de courant initiale à l'aide de l'équation (2)"""
    vort = np.zeros((N, M))
    psi_0 = psi_init()
    for i in range(N):
        # print("i = ", i)
        for j in range(M):
            # print("j = ", j)
            if i == 0:
                # conditions aux bords périodiques : le Nord du domaine = le Sud du domaine
                vort[0, :] = vort[-1, :]
            elif j == 0:
                # conditions aux bords périodiques : l'Ouest du domaine = l'Est du domaine
                vort[:, 0] = vort[:, -1]
            elif i == (N-1):
                # conditions aux bords périodiques : le Sud du domaine = le Nord du domaine
                vort[N-1, :] = vort[0, :]
            elif j == (M-1):
                # conditions aux bords périodiques : l'Est du domaine = l'Ouest du domaine
                vort[:, M-1] = vort[:, 0]
            else:
                vort[i, j] = (1/(Delta_s**2))*(-4*psi_0[i, j]+psi_0[i+1, j] +
                                               psi_0[i-1, j]+psi_0[i, j+1]+psi_0[i, j-1])
                # print("vort({},{}) =".format(i, j), vort[i, j])
    return vort


def velocity_field():
    """Donne champ de vitesse à partir de la fonction de courant"""
    psi_0 = psi_init()
    u = np.zeros((N, M))
    v = np.zeros((N, M))
    for i in range(N):
        # print("i = ", i)
        for j in range(M):
            # print("j = ", j)
            if i == (N-1):
                # conditions aux bords périodiques : le Sud du domaine = le Nord du domaine
                u[N-1, :] = u[0, :]
                v[N-1, :] = v[0, :]
            elif j == (M-1):
                # conditions aux bords périodiques : l'Est du domaine = l'Ouest du domaine
                u[:, M-1] = u[:, 0]
                v[:, M-1] = v[:, 0]
            else:
                u[i, j] = (-1/(2*Delta_s))*(psi_0[i, j+1]-psi_0[i, j-1])
                v[i, j] = (1/(2*Delta_s))*(psi_0[i+1, j]-psi_0[i-1, j])
    return [u, v]


###################################################### Résultats et figures ############################################################


# Output de la résolution des mailles
print("\n\nRésolution numérique avec une grille spatiale de {} points".format(np.shape(x_grid)[1]))
print("Résolution numérique avec une grille temporelle de {} points".format(len(t_grid)))

# Contourplot de la fonction de courant initiale

X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, psi_init(), 100)
plt.colorbar()
plt.title("Contour plot de la fonction de courant initiale $\psi_0(x,y)$ \n $L_x = {}, L_y = {}, \Delta_s = {}, W_x = {}, W_y = {}$".format(
    L_x, L_y, Delta_s, W_x, W_y), fontsize=16)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.show()

# Contourplot de la composante verticale de la vorticité relative initiale
plt.contourf(X, Y, vort_init(), 100)
plt.colorbar()
plt.title("Contour plot de $\zeta_0(x,y)$ \n $L_x = {}, L_y = {}, \Delta_s = {}, W_x = {}, W_y = {}$".format(
    L_x, L_y, Delta_s, W_x, W_y), fontsize=16)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.show()


# Plot du champ de vitesse initial

U = velocity_field()[0]
V = velocity_field()[1]
slice_interval = 1  # Slicer index for smoother quiver function
skip = (slice(None, None, slice_interval), slice(None, None, slice_interval))
vels = np.hypot(U, V)  # Velocity norm of each velocity vector
Quiver = plt.quiver(X[skip], Y[skip], U[skip], V[skip], vels[skip],
                    units='height', angles='xy', scale=350000)
plt.colorbar(Quiver)
# plt.quiverkey(Quiver, 1.01, 1.01, 30000, label="?m/s",
# labelcolor = 'blue', labelpos = 'N', coordinates = "axes")
plt.title("Champ de vitesse initial $(u_0(x,y),v_0(x,y))$ \n $L_x = {}, L_y = {}, \Delta_s = {}, W_x = {}, W_y = {}$".format(
    L_x, L_y, Delta_s, W_x, W_y), fontsize=16)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.show()
