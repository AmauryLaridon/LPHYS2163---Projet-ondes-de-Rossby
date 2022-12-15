import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


###################################################### Paramètres de la simulation ##################################################

phi_0 = ((2*np.pi)/360)*45  # latitude en radian
Delta_s = 200000  # résolution de la maille spatiale en mètre
Delta_t = 360  # résolution de la maille temporelle en seconde
nbr_jour = 1  # nombre de jours de simulation
T = 86400*nbr_jour  # temps total de simulation en seconde
L_x = 12000000  # longueur du domaine selon x en mètre
L_y = 6000000  # longueur du domaine selon y en mètre
M = int(L_x/Delta_s)  # nombre d'itération selon x
N = int(L_y/Delta_s)  # nombre d'itération selon y
K = int(T/Delta_t)  # nombre d'itération selon t
W_x = 6000000  # longueurs d'onde champ initial selon x en mètre
W_y = 3000000  # longueurs d'onde champ initial selon y en mètre
Omega = 7.292115*10**(-5)  # vitesse angulaire de la rotation de la Terre
g = 9.81  # norme de l'accélaration gravitationnelle
a = 6371000  # rayon moyen de la Terre en mètres

###################################################### Discrétisation des mailles ##################################################

x = np.arange(0, L_x, Delta_s)  # discrétisation de l'axe horizontal
y = np.arange(0, L_y, Delta_s)  # discrétisation de l'axe vertical
x_grid = np.matrix(x,)
y_grid = np.matrix(y,)
t_grid = np.arange(0, T, Delta_t)  # discrétisation de l'axe temporel

###################################################### Fonctions de la simulation ##################################################


# Fonctions de la simulation

def f_0_scal():
    """Donne la valeur du paramètre de Coriolis f_0 en un scalaire à phi_0 fixé"""
    return 2*Omega*np.sin(phi_0)


def beta_scal(phi):
    """Donne la valeur du paramètre beta pour une valeur de latitude phi donnée"""
    return (2*Omega*np.cos(phi)/a)


def psi_init():
    """Donne une condition initiale pour la fonction psi_0"""
    k = (2*np.pi)/W_x
    j = (2*np.pi)/W_y
    psi_0_T = np.zeros((N, M))
    psi_0_T = (g/f_0_scal())*(100*np.sin(k*(x_grid.T))@(np.cos(j*y_grid)))
    psi_0 = psi_0_T.T
    return psi_0


def vort_init():
    """Donne la composante verticale de la vorticité relative en fonction de la fonction de courant initiale à l'aide de l'équation(2)"""
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
                vort[i, j] = (1/(Delta_s**2))*(-4*psi_0[i, j]+psi_0[i, j+1] +
                                               psi_0[i, j-1]+psi_0[i+1, j]+psi_0[i-1, j])
                # print("vort({},{}) =".format(i, j), vort[i, j])
    return vort


def velocity_field(psi):
    """Donne champ de vitesse à partir de la fonction de courant"""
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
                u[i, j] = (-1/(2*Delta_s))*(psi[i+1, j]-psi[i-1, j])
                v[i, j] = (1/(2*Delta_s))*(psi[i, j+1]-psi[i, j-1])
    return [u, v]


def vort_flux(u, v, zeta):
    """Donne la valeur du flux de la composante verticale de la vorticité relative F. Prend comme argument le champ de vitesse.
    Prend comme argument la matrice de la composante selon x du champ de vitesse, puis la matrice de la composante y du champs de vitesse
    , la matrice de la composante verticale de la vorticité relative."""
    F = np.zeros((N, M))
    for i in range(N):
        # print("i = ", i)
        for j in range(M):
            # print("j = ", j)
            if i == (N-1):
                # conditions aux bords périodiques : le Sud du domaine = le Nord du domaine
                F[N-1, :] = F[0, :]
            elif j == (M-1):
                # conditions aux bords périodiques : l'Est du domaine = l'Ouest du domaine
                F[:, M-1] = F[:, 0]
            else:
                F[i, j] = (1/(2*Delta_s))*((zeta[i, j+1]*u[i, j+1]-zeta[i, j-1]*u[i, j-1]
                                            ) + (zeta[i+1, j]*v[i+1, j] - zeta[i-1, j]*v[i-1, j])) + beta_scal(phi_0)*v[i, j]
    return F


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
# print(A)
A_inv = np.linalg.inv(A)
# print(A_inv)


def psi(zeta):
    """Fonction qui permet de calculer la fonction de courant psi à partir de la vorticité relative zeta en inversant l'opérateur du laplacien.
    On transforme par différence finie l'opération du laplacien en une matrice A qui donne un ensemble de NxM équations algébriques à NxM inconnes qu'on résoud
    en inversant A."""
    zeta_col = np.zeros((N*M, 1)
                        )  # On met la matrice zeta en forme de colonne en parcourant de gauche à droite et de bas en haut.

    for i in range(N):
        for l in range(M):
            zeta_col[l+i*M, 0] = zeta[i, l]

    """
	Ici intervient notre matrice A définie plus haut.

	En définissant psi_col par rapport au tableau psi de la même manière que l'on a définit zeta_col par rapprot à zeta
	notre système est donc donné par A psi_col = delta_s^2 zeta_col et sa solution est trouvée en inversant la matrice A
	psi_col = A_inv delta_s^2 zeta_col.
	"""
    psi_col = (Delta_s)**2 * np.dot(A_inv, zeta_col)

    # Pour finir il faut remettre psi en forme de tableau pour que ce soit cohérent avec le reste de l'implémentation.
    psi = np.zeros((N, M))
    for i in range(N):
        for l in range(M):
            psi[i, l] = psi_col[l+i*M, 0]

    return psi


###################################################### Intégration temporelle ############################################################


def vort_dynamic():
    # On crée des tableaux à trois dimensions pour les deux dimensions spatiales et une dimension temporelle.
    stream_func_dyn = np.zeros((N, M, K))
    U = np.zeros((N, M, K))
    V = np.zeros((N, M, K))
    vort_dyn = np.zeros((N, M, K))
    F_dyn = np.zeros((N, M, K))
    for t in range(K-1):
        if t == 0:  # On pose les conditions initiale à t=0
            stream_func_dyn[:, :, 0] = psi_init()
            U[:, :, 0] = velocity_field(psi_init())[0]
            V[:, :, 0] = velocity_field(psi_init())[1]
            vort_dyn[:, :, 0] = vort_init()
            F_dyn[:, :, 0] = vort_flux(U[:, :, 0], V[:, :, 0], vort_dyn[:, :, 0])

        else:  # On résoud dans le temps avec le schéma d'Euler avant.
            stream_func_dyn[:, :, t] = psi(vort_dyn[:, :, -1])
            U[:, :, t] = velocity_field(stream_func_dyn[:, :, t])[0]
            V[:, :, t] = velocity_field(stream_func_dyn[:, :, t])[1]
            F_dyn[:, :, t] = vort_flux(U[:, :, t], V[:, :, t], vort_dyn[:, :, t])
            vort_dyn[:, :, t+1] = F_dyn[:, :, t]*Delta_t + vort_dyn[:, :, t]
            # print(vort_dyn)
    return vort_dyn, U, V, stream_func_dyn


###################################################### Résultats et figures ############################################################
# Output de la résolution des mailles
print("-----------------------------------------------------------------------------------------------")
print("Résolution numérique avec une maille spatiale de {}x{} points".format(
    M, N))
print("Résolution numérique avec une maille temporelle de {} points".format(K))
print("-----------------------------------------------------------------------------------------------")

########## Contourplot de la fonction de courant initiale ############
"""
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, psi_init(), 100)
plt.colorbar()
plt.title("Contour plot de la fonction de courant initiale $\psi_0(x,y)$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
    int(L_x/1000), int(L_y/1000), int(Delta_s/1000), int(W_x/1000), int(W_y/1000)), fontsize=11)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
# plt.show()

######### Contourplot de la composante verticale de la vorticité relative initiale ##########
plt.contourf(X, Y, vort_init(), 100)
plt.colorbar()
plt.title("Contour plot de $\zeta_0(x,y)$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
    int(L_x/1000), int(L_y/1000), int(Delta_s/1000), int(W_x/1000), int(W_y/1000)), fontsize=11)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
# plt.show()


########### Plot du champ de vitesse initial ############

U = velocity_field(psi_init())[0]
V = velocity_field(psi_init())[1]
zero = np.zeros((N, M))
slice_interval = 1  # Slicer index for smoother quiver function
skip = (slice(None, None, slice_interval), slice(None, None, slice_interval))
vels = np.hypot(U, V)  # Velocity norm of each velocity vector
Quiver = plt.quiver(X[skip], Y[skip], U[skip], V[skip], vels[skip],
                    units='height', angles='xy')
plt.colorbar(Quiver)
# plt.quiverkey(Quiver, 1.01, 1.01, 30000, label="?m/s",
# labelcolor = 'blue', labelpos = 'N', coordinates = "axes")
plt.title("Champ de vitesse initial $(u_0(x,y),v_0(x,y))$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
    int(L_x/1000), int(L_y/1000), int(Delta_s/1000), int(W_x/1000), int(W_y/1000)), fontsize=11)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
# plt.show()

################################ Plot et animation solution intégrée dans le temps ###################################
"""
solution = vort_dynamic()
vort_dyn = solution[0]
U = solution[1]
V = solution[2]
stream_func_dyn = solution[3]

X, Y = np.meshgrid(x, y)

###################### Plot dynamique animé du champ de vitesse ########################
fig, ax = plt.subplots(1, 1)
u_anim = U[:, :, 0]
v_anim = V[:, :, 0]
Q = ax.quiver(X, Y, u_anim, v_anim, pivot='mid', color='b')


def update_quiver(num, Q, x, y):
    if num == K:
        plt.pause(100)
    print(num)
    u_anim = U[:, :, num]
    v_anim = V[:, :, num]
    nbr_heures = (num*Delta_t)/60
    plt.title('t = {} heures'.format(num*nbr_heures))

    Q.set_UVC(u_anim, v_anim)
    return Q,


anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y), interval=50, blit=False)
fig.tight_layout()
plt.show()

"""

############ Subplot des intégrations temporelles version non animée et pas quali #########
solution = vort_dynamic()
vort_dyn = solution[0]
U = solution[1]
V = solution[2]
stream_func_dyn = solution[3]

X, Y = np.meshgrid(x, y)


for t in range(K):
    fig = plt.figure(figsize=[16/1.3, 9/1.3])
    ax_stream_func = plt.subplot2grid((2, 2), (0, 0))
    ax_U = plt.subplot2grid((2, 2), (0, 1))
    ax_V = plt.subplot2grid((2, 2), (1, 1))
    ax_vort = plt.subplot2grid((2, 2), (1, 0))
    print("itérations = ", t)
    nbr_heures = (t*Delta_t)/60
    nbr_jours = nbr_heures/24
    plt.suptitle("Temps : t = {:.2f} jours".format(nbr_jours))
    ax_stream_func.contourf(X, Y, stream_func_dyn[:, :, t], 100)
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

######################## Début test animation cfr vidéo youtube ####################
fig = plt.figure(figsize=[16/1.3, 9/1.3])
ax_stream_func = plt.subplot2grid((2, 2), (0, 0))
ax_U = plt.subplot2grid((2, 2), (0, 1))
ax_V = plt.subplot2grid((2, 2), (1, 1))
ax_vort = plt.subplot2grid((2, 2), (1, 0))

#fig, ax_stream_func = plt.subplot()


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
