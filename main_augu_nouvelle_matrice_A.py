import numpy
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.pylab import *
import numpy as np
import random as rand
import math
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

###################################################### Paramètres de la simulation ##################################################
phi_0 = (2*np.pi/360)*45  # latitude en radian par défaut la latitude est de 45°
Lx = 12000000  # longueur de la maille spatiale. Par défaut 12 000km
Ly = 6000000  # larger de la maille spatiale. Par défaut 6 000km
Wx = 6000000  # longueurs d'onde champ initial selon x en mètre. Par défaut 6 000km
Wy = 3000000  # longueurs d'onde champ initial selon y en mètre. Par défaut 3 000km
delta_s = 200000  # résolution de la maille spatiale en mètre. Valeur par défaut = 200km
delta_t_en_heure = 1
delta_t = 3600 * delta_t_en_heure  # passage au SI
temps_integration_en_jour = 12
temps_integration_en_seconde = 60 * 60 * 24 * temps_integration_en_jour  # passage au SI
M = int(Lx/delta_s)  # nombre d'itération selon x
N = int(Ly/delta_s)  # nombre d'itération selon y
nb_pas_de_temps = int(temps_integration_en_seconde/delta_t)
rayon_terre = 6371000  # rayon moyen de la Terre en mètres
g = 9.81  # norme de l'accélération gravitationnelle
omega = 7.29215*10**(-5)  # vitesse angulaire de la rotation de la Terre
save_count_value_T_12 = 286*(1/delta_t_en_heure)  # variable définis temps sauvegarde animation.
enregistrement_pas_affich = True  # variable définis si on enregistre si on affiche les résultats.

###################################################### Discrétisation des mailles ##################################################
# Crée deux tableaux de taille NxM, l'un avec les valeurs discrétisée de x et l'autre de y
xvalues, yvalues = np.meshgrid(np.arange(0, Lx, delta_s), np.arange(0, Ly, delta_s))
###################################################### Fonctions de la simulation ##################################################
"""
Note sur la périodicité:
	Les modules (%) sont utilisés pour faire la périodicité aux bords. Ainsi pour un psi situé par exemple sur le bord est il prendra en compte pour son calcul
son voisin fictif situé sur le bord ouest.
"""


def f_0_scal(phi_0):
    """Donne la valeur du paramètre de Coriolis f_0 en un scalaire à phi_0 fixé"""
    return (2*omega*np.sin(phi_0))


def beta_scal(phi):
    """Donne la valeur du paramètre beta pour une valeur de latitude phi donnée"""
    return (2*omega*np.cos(phi)/rayon_terre)


def psi_init():
    """Donne une condition initiale pour la fonction psi_0"""
    k = (2*np.pi)/Wx
    j = (2*np.pi)/Wy
    x = np.arange(0, Lx, delta_s)  # discrétisation de l'axe horizontal
    y = np.arange(0, Ly, delta_s)  # discrétisation de l'axe vertical
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
            zeta[i, l] = (1/(delta_s**2)) * (-4 * psi[i, l] + psi[(i+1) %
                                                                  N, l] + psi[i-1, l] + psi[i, (l+1) % M] + psi[i, l-1])

    return zeta


def u(psi):
    """Donne la composante zonale du champ de vitesse à partir de la fonction de courant"""
    u = np.zeros((N, M))
    for i in range(N):
        for l in range(M):
            u[i, l] = (-1/(2*delta_s))*(psi[(i+1) % N, l] - psi[i-1, l])
    return u


def v(psi):
    """Donne la composante méridienne du champ de vitesse à partir de la fonction de courant"""
    v = np.zeros((N, M))
    for i in range(N):
        for l in range(M):
            v[i, l] = (1/(2*delta_s))*(psi[i, (l+1) % M] - psi[i, l-1])

    return v


def vort_flux(zeta, u, v):
    """Donne la valeur du flux de la composante verticale de la vorticité relative F. Prend comme argument le champ de vitesse.
Prend comme argument la matrice de la composante selon x du champ de vitesse, puis la matrice de la composante y du champs de vitesse
, la matrice de la composante verticale de la vorticité relative."""
    vort_flux = np.zeros((N, M))
    for i in range(N):
        for l in range(M):
            vort_flux[i, l] = (1/(2*delta_s))*(zeta[i, (l+1) % M]*u[i, (l+1) % M] - zeta[i, l-1] *
                                               u[i, l-1] + zeta[(i+1) % N, l]*v[(i+1) % N, l] - zeta[i-1, l]*v[i-1, l]) + beta_scal(phi_0) * v[i, l]

    return vort_flux


def zeta(F, zeta):
    """Donne la valeur de la vorticité à un instant t à partir à du flux de vorticité et la valeur de la vorticité au temps t-delta_t.
    Sera utile dans l'intégration temporelle"""
    new_zeta = np.zeros((N, M))
    for i in range(N):
        for l in range(M):
            new_zeta[i, l] = -F[i, l] * 2 * delta_t + zeta[i, l]
    return new_zeta


"""
	Ici on définit une grande matrice A qui est utile pour la fonction psi(zeta) utilisée juste après. On la définit ici car
	la fonction sera utilisée de nombreuse fois et recrée cette matrice et l'inverser à chaque fois prendrait beaucoup de temps
	de calcul pour rien. Notez que les commentaires ajoutés au cours de la création de la matrice sont à comprendre
	dans le contexte de la fonction psi(zeta) ci-dessous
"""

### Output de la résolution des mailles ###
print("-----------------------------------------------------------------------------------------------")
print("Résolution numérique avec une maille spatiale de {}x{} points".format(
    M, N))
print("Résolution numérique avec une maille temporelle de {} points".format(
    temps_integration_en_seconde))
print("-----------------------------------------------------------------------------------------------")

# Définition d'une grande matrice carrée diagonale A de côté N x M avec -4 sur sa diagonale.
A = -4*np.eye(N*M)
"""
On veut sur chaque ligne ajouter un terme "1" correspondant à l'emplacement des 4 valeurs adjacentes de psi_i,j.
"""
col = 0
print("Calcul matrice A de l'opérateur Laplacien")
print("------------------------------------------")
for lin_A in range(N*M):
    if lin_A in np.arange(M, N*M, M):
        col += 1
    lin = lin_A % M
    col_courant = 0
    print('ligne {} / {}'.format(lin_A, N*M))
    for col_A in range(N*M):

        if col_A in np.arange(M, N*M, M):
            col_courant += 1
        lin_courant = col_A % M

        if col_courant == col and lin_courant == (lin+1) % M:
            A[lin_A, col_A] = 1

        if col_courant == col and lin_courant == (lin-1) % M:
            A[lin_A, col_A] = 1

        if col_courant == (col+1) % N and lin_courant == lin:
            A[lin_A, col_A] = 1

        if col_courant == (col-1) % N and lin_courant == lin:
            A[lin_A, col_A] = 1
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

    """
	Ici intervient notre matrice A définie plus haut.
	En définissant psi_col par rapport au tableau psi de la même manière que l'on a définit zeta_col par rapprot à zeta
	notre système est donc donné par A psi_col = delta_s^2 zeta_col et sa solution est trouvée en inversant la matrice A
	psi_col = A_inv delta_s^2 zeta_col.
	"""
    psi_col = delta_s**2 * np.dot(A_inv, zeta_col)
    # Pour finir il faut remettre psi en forme de tableau pour que ce soit cohérent avec le reste de l'implémentation.
    psi = np.zeros((N, M))
    for i in range(N):
        for l in range(M):
            psi[i, l] = psi_col[l+i*M, 0]
    return psi

################################################## - Intégration Numérique - ###############################################


def zeta_dynamic():
    # à chaque itération du temps le résultat sera ajouté en dernière place de ces vecteurs.
    nb_plot = 1
    for t in range(nb_pas_de_temps):
        if t == 0:  # On pose les CI
            u_tot = [u_0]
            v_tot = [v_0]

            F_tot = [vort_flux(zeta_0, u_0, v_0)]
            zeta_tot = [zeta_0]

            zeta_tot.append(np.zeros((N, M)))
            for i in range(N):
                for l in range(M):
                    zeta_tot[-1][i, l] = - F_tot[-1][i, l] * delta_t + zeta_0[i, l]

            psi_tot = [psi(zeta_tot[-1])]
            nbr_jours = t*delta_t_en_heure/24
            print("---------------------------------------------------")
            print("Itérations = ", 1, "/", nb_pas_de_temps)
            print("Temps : t = {:.2f} heures = {:.2f} jours".format(0, (t*delta_t_en_heure)/24))
        else:
            u_tot.append(u(psi_tot[-1]))
            v_tot.append(v(psi_tot[-1]))  # Ajout d’un écoulement moyen
            F_tot.append(vort_flux(zeta_tot[-1], u_tot[-1], v_tot[-1]))
            zeta_tot.append(zeta(F_tot[-1], zeta_tot[-2]))
            psi_tot.append(psi(zeta_tot[-1]))
            nbr_heures = t*delta_t_en_heure
            print("---------------------------------------------------")
            print("Itérations = ", t+1, "/", nb_pas_de_temps)
            print("Temps : t = {:.2f} heures = {:.2f} jours".format(
                nbr_heures, (t*delta_t_en_heure)/24))
    print("---------------------------------------------------")
    return u_tot, v_tot, zeta_tot, psi_tot, F_tot

############################################### - Affichage Conditions Initiales ############################################
########## Contourplot de la fonction de courant initiale ############


def contour_plot_psi0():
    plt.contourf(xvalues, yvalues, psi_0, 100)
    plt.colorbar()
    plt.title("Contour plot de la fonction de courant initiale $\psi_0(x,y)$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
        int(Lx/1000), int(Ly/1000), int(delta_s/1000), int(Wx/1000), int(Wy/1000)), fontsize=11)
    plt.legend(loc='best', shadow=True, fontsize="large")
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("$y$", fontsize=20)
    plt.tight_layout()
    plt.show()
#### Contourplot de la composante verticale de la vorticité relative initiale ###


def contour_plot_zeta_0():
    plt.contourf(xvalues, yvalues, zeta_0, 100)
    plt.colorbar()
    plt.title("Contour plot de $\zeta_0(x,y)$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
        int(Lx/1000), int(Ly/1000), int(delta_s/1000), int(Wx/1000), int(Wy/1000)), fontsize=11)
    plt.legend(loc='best', shadow=True, fontsize="large")
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("$y$", fontsize=20)
    plt.tight_layout()
    plt.show()
######################## Plot du champ de vitesse initial ######################


def quiver_velocity_field():
    slice_interval = 1  # Slicer index for smoother quiver function
    skip = (slice(None, None, slice_interval), slice(None, None, slice_interval))
    vels = np.hypot(u_0, v_0)  # Velocity norm of each velocity vector
    Quiver = plt.quiver(xvalues[skip], yvalues[skip], u_0[skip], v_0[skip], vels[skip],
                        units='height', angles='xy')
    plt.colorbar(Quiver)
    # plt.quiverkey(Quiver, 1.01, 1.01, 30000, label="?m/s",
    # labelcolor = 'blue', labelpos = 'N', coordinates = "axes")
    plt.title("Champ de vitesse initial $(u_0(x,y),v_0(x,y))$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
        int(Lx/1000), int(Ly/1000), int(delta_s/1000), int(Wx/1000), int(Wy/1000)), fontsize=11)
    plt.legend(loc='best', shadow=True, fontsize="large")
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("$y$", fontsize=20)
    plt.tight_layout()
    plt.show()

######################## Subplot champs initiaux ######################


def init_subplot():

    fig = plt.figure(figsize=[16/1.3, 9/1.3])
    plt.gcf().subplots_adjust(left=0.05, bottom=0.08, right=0.93, top=0.9, wspace=0.22, hspace=0.28)
    ax_stream_func = plt.subplot2grid((2, 2), (0, 0))
    ax_U = plt.subplot2grid((2, 2), (0, 1))
    ax_V = plt.subplot2grid((2, 2), (1, 1))
    ax_vort = plt.subplot2grid((2, 2), (1, 0))
    plt.suptitle("Temps : t = {:.2f} heures = {:.2f} jours".format(0, 0))

    im01 = ax_stream_func.contourf(xvalues, yvalues, psi_tot[0], 1000)
    ax_stream_func.set_title("$\psi(x,y,t)$")

    divider = make_axes_locatable(ax_stream_func)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    fig.colorbar(im01, cax=cax, orientation='vertical')

    im02 = ax_U.contourf(xvalues, yvalues, u_tot[0], 1000)
    ax_U.set_title("$u(x,y,t)$")

    divider = make_axes_locatable(ax_U)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    fig.colorbar(im02, cax=cax, orientation='vertical')

    im03 = ax_V.contourf(xvalues, yvalues, v_tot[0], 1000)
    ax_V.set_title("$v(x,y,t)$")
    ax_V.set_xlabel("$x$")
    ax_V.set_ylabel("$y$")

    divider = make_axes_locatable(ax_V)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    fig.colorbar(im03, cax=cax, orientation='vertical')

    im04 = ax_vort.contourf(xvalues, yvalues, zeta_tot[0], 1000)
    ax_vort.set_title("$\zeta(x,y,t)$")
    ax_vort.set_xlabel("$x$")
    ax_vort.set_ylabel("$y$")

    divider = make_axes_locatable(ax_vort)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    fig.colorbar(im04, cax=cax, orientation='vertical')

    plt.show()


################################################ Affichage Intégration Numérique #############################################
############# Plot dynamique des vecteurs vitesses au cours du temps ####################


def dyn_plot_velocity_field():
    u_anim = u_tot[0]
    v_anim = v_tot[0]
    vels = np.hypot(u_0, v_0)
    fig, ax = plt.subplots(1, 1)
    Q = ax.quiver(xvalues, yvalues, u_anim, v_anim, vels, pivot='mid', color='r')
    plt.colorbar(Q)

    def update_quiver(t, Q, x, y):
        if t == (nb_pas_de_temps-1):
            plt.pause(100)
        u_anim = u_tot[t]
        v_anim = v_tot[t]
        #vector_anim_max_vel = np.hypot(u_anim, v_anim)

        # print(vector_anim_max_vel)
        title("$(u(x,y,t), v(x,y,t)), t ={:.2f}h, jours = {}$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ \n $T = {}\;  jours, \Delta_t = {}h $ ".format(
            t*(delta_t/3600), int(t*delta_t_en_heure/24), int(Lx/1000), int(Ly/1000), int(delta_s/1000), int(Wx/1000), int(Wy/1000), int(temps_integration_en_jour), int(delta_t/3600)), fontsize=8)
        xlabel("$x$")
        ylabel("$y$")
        plt.tight_layout()
        Q.set_UVC(u_anim, v_anim)
        return Q,

    # you need to set blit=False, or the first set of arrows never gets
    # cleared on subsequent frames
    ani = animation.FuncAnimation(fig, update_quiver, fargs=(
        Q, xvalues, yvalues), interval=200, blit=False, save_count=save_count_value_T_12)  # save_count = max 286 pour T = 12 jours avec delta_t = 1h, save_count = max 572 pour T = 12 jours et delta_t = 0.5h
    # fig.tight_layout()

    if enregistrement_pas_affich == True:
        # paramètres objet writers
        Writer = writers['ffmpeg']
        writer = Writer(fps=10, bitrate=-1)
        ani.save('dyn_plot_velocity_field.mp4', writer)
    else:
        plt.show()

#################### Plot dynamique de zeta_dyn au cours du temps #################


def dyn_plot_zeta():
    fig = plt.figure()
    im = plt.imshow(zeta_tot[0], interpolation='nearest', cmap='Blues')
    colorbar()

    def update1(data):
        im.set_array(data)

    def data_gen_zeta(n):
        for t in range(n):
            title("$\zeta(x,y,t), t ={:.2f}h, jours = {}$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ \n $T = {}\;  jours, \Delta_t = {}h $ ".format(
                t*(delta_t/3600), int(t*delta_t_en_heure/24), int(Lx/1000), int(Ly/1000), int(delta_s/1000), int(Wx/1000), int(Wy/1000), int(temps_integration_en_jour), int(delta_t/3600)), fontsize=8)
            xlabel("$x$")
            ylabel("$y$")
            plt.tight_layout()
            yield zeta_tot[t]

    ani = animation.FuncAnimation(fig, update1, data_gen_zeta(
        nb_pas_de_temps), interval=200, save_count=save_count_value_T_12)

    if enregistrement_pas_affich == True:
        # paramètres objet writers
        Writer = writers['ffmpeg']
        writer = Writer(fps=5, metadata={'artist': 'Me'}, bitrate=-1)
        ani.save('dyn_plot_zeta.mp4', writer)
    else:
        plt.show()
#################### Plot dynamique de psi_dyn au cours du temps #################


def dyn_plot_psi():
    fig = plt.figure()

    im = plt.imshow(psi_tot[0], interpolation='nearest', cmap='Blues')
    colorbar()

    def update(data):
        im.set_array(data)

    def data_gen(n):
        for n in range(n):
            title("$\psi(x,y,t), t ={:.2f}h, jours = {}$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ \n $T = {}\;  jours, \Delta_t = {}h $ ".format(
                n*(delta_t/3600), int(n*delta_t_en_heure/24), int(Lx/1000), int(Ly/1000), int(delta_s/1000), int(Wx/1000), int(Wy/1000), int(temps_integration_en_jour), int(delta_t/3600)), fontsize=10)
            xlabel("$x$")
            ylabel("$y$")
            plt.tight_layout()
            yield psi_tot[n]

    ani = animation.FuncAnimation(fig, update, data_gen(
        nb_pas_de_temps), interval=200, save_count=save_count_value_T_12)

    if enregistrement_pas_affich == True:
        # paramètres objet writers
        Writer = writers['ffmpeg']
        writer = Writer(fps=5, metadata={'artist': 'Me'}, bitrate=-1)
        name_file = 'dyn_plot_psi_delta_s_{}_delta_t_{}.mp4'.format(delta_s, delta_t)
        ani.save('dyn_plot_psi.mp4', writer)
    else:
        plt.show()

#################### SubPlot dynamique des champs au cours du temps #################


def dyn_subplot():
    ### V1.0 non opérationnelle subplot_dyn() ###
    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex='col', sharey='row')

    # fig = plt.figure()

    im1 = axs[0, 0].imshow(psi_tot[0], cmap='Blues', animated=True)

    im2 = axs[0, 1].imshow(u_tot[0], cmap='Blues', animated=True)

    im3 = axs[1, 0].imshow(zeta_tot[0], cmap='Blues', animated=True)

    im4 = axs[1, 1].imshow(v_tot[0], cmap='Blues', animated=True)

    def update(data):
        im1.set_array(data)
        im2.set_array(data)
        im3.set_array(data)
        im4.set_array(data)

    def data_gen(n):
        for n in range(n):
            # title("$\psi(x,y,t), t ={:.2f}h, jours = {}$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ \n $T = {}\;  jours, \Delta_t = {}h $ ".format(
            # n*(delta_t/3600), int(n*delta_t_en_heure/24), int(Lx/1000), int(Ly/1000), int(delta_s/1000), int(Wx/1000), int(Wy/1000), int(temps_integration_en_jour), int(delta_t/3600)), fontsize=10)
            # xlabel("$x$")
            # ylabel("$y$")
            # plt.tight_layout()
            yield psi_tot[n]
            yield u_tot[n]
            yield zeta_tot[n]
            yield v_tot[n]

    ani = animation.FuncAnimation(fig, update, data_gen(
        nb_pas_de_temps), interval=200, save_count=save_count_value_T_12)
    plt.show()
### V2.0 non opérationnelle subplot_dyn() ###


"""
    # Sent for figure
    font = {'size': 9}
    matplotlib.rc('font', **font)

    # Setup figure and subplots
    f0 = figure(num=0, figsize=(12, 8))  # , dpi = 100)
    f0.suptitle("Intégration numérique", fontsize=12)
    ax01 = subplot2grid((2, 2), (0, 0))
    ax02 = subplot2grid((2, 2), (0, 1))
    ax03 = subplot2grid((2, 2), (1, 0))
    ax04 = subplot2grid((2, 2), (1, 1))
    # tight_layout()

    # Set titles of subplots
    ax01.set_title('$\psi(x,y,t)$')
    ax02.set_title('$u(x,y,t)$')
    ax03.set_title('$\zeta(x,y,t)$')
    ax04.set_title('$v(x,y,t)$')

    # set label names
    ax01.set_xlabel("x")
    ax01.set_ylabel("y")
    ax02.set_xlabel("x")
    ax02.set_ylabel("y")
    ax03.set_xlabel("x")
    ax03.set_ylabel("y")
    ax04.set_xlabel("x")
    ax04.set_ylabel("y")

    # Data Placeholders
    t = zeros(0)
    # set plots
    im1 = ax01.imshow(psi_tot[0], cmap='Blues', animated=True)

    im2 = ax02.imshow(u_tot[0], cmap='Blues', animated=True)

    im3 = ax03.imshow(zeta_tot[0], cmap='Blues', animated=True)

    im4 = ax04.imshow(v_tot[0], cmap='Blues', animated=True)

    def updateData(t):

        im1.set_array(t)
        im2.set_array(t)
        im3.set_array(t)
        im4.set_array(t)

        return im1, im2, im3, im4

    # interval: draw new frame every 'interval' ms
    # frames: number of frames to draw
    simulation = animation.FuncAnimation(
        f0, updateData(nb_pas_de_temps), blit=False, frames=200, interval=20, repeat=False)

    # Uncomment the next line if you want to save the animation
    # simulation.save(filename='sim.mp4',fps=30,dpi=300)

    plt.show()

"""
if __name__ == "__main__":
    ### Conditions Initiales ###
    psi_0 = psi_init()
    zeta_0 = zeta_init(psi_0)
    u_0 = u(psi_0)
    v_0 = v(psi_0)
    #### Récupération des résultats ####
    solution = zeta_dynamic()
    u_tot = solution[0]
    v_tot = solution[1]
    zeta_tot = solution[2]
    psi_tot = solution[3]
    ### Affichage Solutions ###
    # contour_plot_psi0()
    # contour_plot_zeta_0()
    # quiver_velocity_field()
    # init_subplot()
    dyn_plot_velocity_field()
    dyn_plot_zeta()
    dyn_plot_psi()
    # dyn_subplot()
