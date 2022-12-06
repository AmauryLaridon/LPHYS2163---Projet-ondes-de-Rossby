import matplotlib.pyplot as plt
import numpy as np


# Paramètres de la simulation

phi_0 = ((2*np.pi)/360)*45  # latitude en radian
Delta_s = 200  # résolution de la maille
L_x = 12000  # longueur du domaine selon x
L_y = 6000  # longueur du domaine selon y en km
M = int(L_x/Delta_s)  # nombre d'itération selon x
N = int(L_y/Delta_s)  # nombre d'itération selon x
W_x = 6000  # longueurs d'onde champ initial selon x
W_y = 3000  # longueurs d'onde champ initial selon y
Omega = 7.292115*10**(-5)  # vitesse angulaire de la rotation de la Terre
g = 9.81  # norme de l'accélaration gravitationnelle

x = np.arange(0, L_x, Delta_s)  # discrétisation de l'axe horizontal
y = np.arange(0, L_y, Delta_s)  # discrétisation de l'axe vertical
x = np.matrix(x,)
y = np.matrix(y,)

# Fonctions de la simulation


def f_0(Omega, phi_0):
    """Donne la valeur du paramètre de Coriolis f_0"""
    return (2*Omega*np.sin(phi_0))


def psi_init():
    """Donne une condition initiale pour la fonction psi_0"""
    k = (2*np.pi)/W_x
    j = (2*np.pi)/W_y
    psi_0_T = np.zeros((N, M))
    #print("taille de x : ", np.shape(x), "taille de y: ", np.shape(y))
    #print("x", x.T)
    #print("y", y)
    # print(x.T@y)
    psi_0_T = (g/f_0(Omega, phi_0))*(100*np.sin(k*(x.T))@(np.cos(j*y)))
    # print(np.shape(psi_0_T))
    psi_0 = psi_0_T.T
    # print(np.shape(psi_0))
    return psi_0


# Figures de la simulation


X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, psi_init(), 100)
plt.colorbar()
plt.title("Contour plot de la fonction de courant initiale $\psi_0(x,y)$ \n $L_x = {}, L_y = {}, \Delta_s = {}, W_x = {}, W_y = {}$".format(
    L_x, L_y, Delta_s, W_x, W_y), fontsize=26)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.show()
