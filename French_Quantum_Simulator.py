import numpy as np
import pygame
import pygame.locals as pg
from pygame.locals import *
from math import *
import os
import subprocess
import tempfile


pygame.init()
fenetre = pygame.display.set_mode([1500, 900])
pygame.display.set_caption("Simulateur quantique")


h = 6.626e-34
c = 3e8
h_barre = h / (2 * np.pi)


image = pygame.image.load(os.path.join("Data", "Zine-Eddine-Daouadi.png"))
image_2 = pygame.image.load(os.path.join("Data", "LinkedIn_Logo.png"))
image_3 = pygame.image.load(os.path.join("Data", "Logo_Github.png"))
image_4 = pygame.image.load(os.path.join("Data", "Blogger_Logo.jpg"))
formule1 = pygame.image.load(os.path.join("Data", "Formules_Quantique.png"))
formule2 = pygame.image.load(os.path.join("Data", "Formules_Quantique_2.png"))
formule3 = pygame.image.load(os.path.join("Data", "Formules_Quantique_3.png"))
formule4 = pygame.image.load(os.path.join("Data", "Formules_Quantiques_4.png"))
Planck = pygame.image.load(os.path.join("Data", "Planck.png"))
Heisenberg = pygame.image.load(os.path.join("Data", "Heisenberg.png"))
Broglie = pygame.image.load(os.path.join("Data", "Broglie.png"))
Janson = pygame.image.load(os.path.join("Data", "Lycée_Janson_De_Sailly_Logo.png"))
Saclay = pygame.image.load(os.path.join("Data", "Université_Paris_Saclay_Logo.png"))
young = pygame.image.load(os.path.join("Data", "young.png"))
pdf = pygame.image.load(os.path.join("Data", "pdf_logo.png"))
Logo = pygame.image.load(os.path.join("Data", "Logo.png"))
pygame.display.set_icon(Logo)

def rayon(x, y, z):
    return sqrt(x**2 + y**2 + z**2)

def longueur_ondes(p):
    return h / p

def densité(A, r):
    return (abs(A) / r)**2

def probabilité(x, y, L, A):
    return (abs(A)**2) / (x**2 + y**2 + L**2)

def probabilité_sphérique(A, r, m, k):
    return (h_barre * k * (abs(A)**2) / (m * r**2))

def sinc2(u):
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.sin(u) / u
        s[np.abs(u) < 1e-10] = 1.0
    return s**2


def test_conique(beta, n):
    return beta * n >= 1

def angle_emission(beta, n):
    return np.arccos(1/(beta*n))

def rayon_final(theta, L):
    return L * np.tan(theta)

def calcul_cherenkov(beta, L, n, nx, ny, largeur_ecran):
    
    x = np.linspace(-largeur_ecran/2, largeur_ecran/2, nx)
    y = np.linspace(-largeur_ecran/2, largeur_ecran/2, ny)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    
    
    dx = largeur_ecran / nx
    
    
    if not test_conique(beta, n):
        print("Condition de seuil non satisfaite : pas d'émission Cherenkov.")
        
        R_fictif = 0.005  
        image = np.zeros((ny, nx))
        mask = np.abs(r - R_fictif) < dx
        image[mask] = 1.0
        return image / image.max()
    
    
    lam_min = 400e-9
    lam_max = 700e-9
    n_lam = 100
    lambdas = np.linspace(lam_min, lam_max, n_lam)
    
    
    image = np.zeros((ny, nx))
    
    
    theta = angle_emission(beta, n)
    R = rayon_final(theta, L)
    
   
    if R > largeur_ecran/2:
        print(f"Attention : l'anneau (R={R*1000:.1f} mm) dépasse l'écran. Ajustez L ou la taille de l'écran.")
        R = largeur_ecran/2 * 0.9  
    
    
    for lam in lambdas:
        facteur = (1.0 - 1.0/(beta**2 * n**2)) / (lam**3)
        mask = np.abs(r - R) < dx
        image[mask] += facteur
    
    
    if image.max() == 0:
        print("Aucun pixel n'a été touché, création d'un anneau par défaut.")
        R_defaut = 0.005
        mask = np.abs(r - R_defaut) < dx
        image[mask] = 1.0
    else:
        image = image / image.max()
    
    return image


def calcul_source_seule_plane(X, Y, L, A, lmbda, **kwargs):
    r = np.sqrt(X**2 + Y**2 + L**2)
    I = np.abs(A)**2 / r**2
    return I

def calcul_fente_unique_plane(X, Y, L, a, b, lmbda, A=1):
    u = np.pi * a * X / (lmbda * L)
    v = np.pi * b * Y / (lmbda * L)
    I = np.abs(A)**2 * sinc2(u) * sinc2(v)
    return I

def calcul_double_fente_plane(X, Y, L, a, b, d, lmbda, A=1):
    I_fente = calcul_fente_unique_plane(X, Y, L, a, b, lmbda, 1)
    interf = np.cos(np.pi * d * X / (lmbda * L))**2
    return np.abs(A)**2 * I_fente * interf

def calcul_reseau_plane(X, Y, L, a, b, d, N, lmbda, A=1):
    I_fente = calcul_fente_unique_plane(X, Y, L, a, b, lmbda, 1)
    alpha = np.pi * d * X / (lmbda * L)
    terme_reseau = (np.sin(N * alpha) / np.sin(alpha))**2
    terme_reseau = np.where(np.abs(np.sin(alpha)) < 1e-10, N**2, terme_reseau)
    return np.abs(A)**2 * I_fente * terme_reseau


def calcul_spherique(X, Y, L, D, a, b, d, N, lmbda, A):
    nx = X.shape[1]
    ny = X.shape[0]
    n_pts_largeur = 5  
    n_pts_hauteur = 5  
    dx_fente = a / n_pts_largeur
    dy_fente = b / n_pts_hauteur
    centres_x = np.linspace(-(N-1)*d/2, (N-1)*d/2, N)
    x_sources = []
    y_sources = []
    for cx in centres_x:
        for i in range(n_pts_largeur):
            x_local = cx + (i - n_pts_largeur//2 + 0.5) * dx_fente
            for j in range(n_pts_hauteur):
                y_local = (j - n_pts_hauteur//2 + 0.5) * dy_fente
                x_sources.append(x_local)
                y_sources.append(y_local)
    x_sources = np.array(x_sources)
    y_sources = np.array(y_sources)
    Ns = len(x_sources)
    print(f"Nombre de points sources pour le calcul sphérique : {Ns}")
    k = 2 * np.pi / lmbda
    psi = np.zeros((ny, nx), dtype=complex)
    for xs, ys in zip(x_sources, y_sources):
        r1 = np.sqrt(xs**2 + ys**2 + D**2)
        phase1 = np.exp(1j * k * r1) / r1
        r2 = np.sqrt((X - xs)**2 + (Y - ys)**2 + L**2)
        phase2 = np.exp(1j * k * r2) / r2
        psi += phase1 * phase2 * (dx_fente * dy_fente)
    psi = A * psi
    I = np.abs(psi)**2
    return I


class InputBox:
    def __init__(self, x, y, w, h, font, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = pygame.Color('black')
        self.color_active = pygame.Color('red')
        self.color = self.color_inactive
        self.font = font
        self.text = text
        self.txt_surface = font.render(text, True, self.color)
        self.active = False

    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive

        if event.type == pg.KEYDOWN and self.active:
            if event.key == pg.K_RETURN:
                pass
            elif event.key == pg.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                if event.unicode.isprintable():
                    new_text = self.text + event.unicode
                    if self.font.size(new_text)[0] <= self.rect.w - 10: 
                        self.text = new_text
            self.txt_surface = self.font.render(self.text, True, self.color)

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect)
        pygame.draw.rect(screen, self.color, self.rect, 2)
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))
        if self.active and (pygame.time.get_ticks() // 500) % 2 == 0:
            cursor_x = self.rect.x + 5 + self.txt_surface.get_width()
            cursor_y = self.rect.y + 5
            pygame.draw.line(screen, self.color, (cursor_x, cursor_y),
                             (cursor_x, cursor_y + self.font.get_height()), 2)

def parse_float_list(text):
    
    if text.strip() == "":
        return []
    parts = text.split(',')
    result = []
    for p in parts:
        try:
            result.append(float(p.strip()))
        except ValueError:
            pass
    return result

def intensite_vers_surface(I, use_colormap=True):
    
    if use_colormap:
        import matplotlib.cm as cm
        I_norm = (I - I.min()) / (I.max() - I.min() + 1e-15)
        rgb = (cm.viridis(I_norm)[:, :, :3] * 255).astype(np.uint8)
    else:
        I_8 = (255 * (I - I.min()) / (I.max() - I.min() + 1e-15)).astype(np.uint8)
        rgb = np.stack((I_8, I_8, I_8), axis=-1)
    # pygame attend (largeur, hauteur) -> échange des axes
    surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    return surface

def sauvegarder_et_ouvrir(surface):
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        filename = tmp.name
    pygame.image.save(surface, filename)
    # Ouvre l'image avec l'application par défaut (sous Windows)
    if os.name == 'nt':
        os.startfile(filename)
    else:
        # Pour Linux/Mac, essayer d'ouvrir avec xdg-open ou open
        subprocess.call(('xdg-open', filename))

# Données textuelles pour l'interface
Liste_options = ["Simulateur de particules.", "Paranthèse Historique.", "Auteur du simulateur.", "Formules Mathématiques.", "Quitter."]
Description_Auteur = [
    "Bonjour, je suis DAOUADI Zine-Eddine, étudiant en MP2I au Lycée Janson-De-Sailly.",
    "J'aime la science, la littérature et la finance.",
    "J'aspire à m'insérer dans la finance ou dans des études de très haut-niveau en Photonique",
    "quantique. Cet amour pour la finance, est donc commun à celui de la Physique.",
    "J'ai réalisé ce simulateur, dans mon temps libre afin de m'exercer mais aussi de m'entraîner",
    "à modéliser.",
    "Le but de ce simulateur est de permettre une visualisation des ondes sphériques et",
    "du rayonnement Cherenkov selon leurs propriétés.",
    "Une certaine liberté est donnée à l'utilisateur, pour échelonner les variables.",
    "J'invite à tester les limites du simulateur, mais aussi apprécier ce en quoi il permet de voir",
    "les limites du modèle quantique.",
    "Je vous souhaite une bonne utilisation, merci d'avoir pris le temps de s'y intéresser !",
    "Cordialement,",
    "DAOUADI Zine-Eddine.",
    "Vous pouvez aussi me retrouver sur :"
]

Formules_Mathematiques = ["Un grand nombre de formules, on été nécessaires afin de modéliser",
                          "le phénomène de répartition sur un écran des tâches.",
                          "Néanmoins, j'aimerais vous donner la liste exhaustive afin",
                          "que vous puissiez non seulement refaire les calculs pour vérifier",
                          "les résultats... mais aussi apprendre !",
                          "J'ai dû faire beaucoup de recherches pour mettre en oeuvre",
                          "au point où à vrai dire, j'ai passé des semaines à rechercher",
                          "intensivement des données.",
                          "Ce simulateur a cependant des limites théoriques.",
                          "On pense à la décohérence, notamment qui augmenterait la complexité",
                          "en temps et en espace, on a alors fait des approximations.",
                          "Le but a été de réaliser un modèle calculatoire efficace.",
                          "D'où une partie des limites qui régissent le programme.",
                          "Vous trouverez néanmoins un pdf avec les formules utilisées",
                          "dans le repository.",
                          "Le but reste néanmoins de fournir une vue d'ensemble, permettant",
                          "à des étudiants de se faire une image des liens entre mécanique",
                          "quantique et particules à travers l'expériences des fentes",
                          "de Young."]

# Paramètres par mode
parametres_par_mode = {
    "Plane": [
        "Masse (kg) :",
        "Distance Z (m) :",
        "Amplitude A :",
        "Nombre de fentes N :",
        "Largeur x (cm) :",
        "Hauteur y (cm) :",
        "Ecart d (cm) :",
        "Energie E_c (J) :"
    ],
    "Sphérique": [
        "Masse (kg) :",
        "Distance Z (m) :",
        "Amplitude A :",
        "Nombre de fentes N :",
        "Largeur x (cm) :",
        "Hauteur y (cm) :",
        "Ecart d (cm) :",
        "Energie E_c (J) :",
        "Distance source-fentes (m) :"
    ],
    "Conique": [
        "Vitesse β (v/c) :",
        "Indice de réfraction n :",
        "Distance écran L (m) :"
    ]
}

Onglet_Quitter = [
    "Un grand merci à vous d'avoir utilisé ce simulateur !",
    "J'aspire à continuer à donner forme au réel par le biais",
    "de la symbiose entre Physique et Informatique.",
    "Je la trouve même nécessaire pour le développement de la science :",
    "la communication étant au centre du progrès.",
    "Je vous souhaite une bonne continuation !",
    "Vous pouvez appuyer sur entrée pour quitter.",
    "Cordialement,",
    "DAOUADI Zine-Eddine"
]

Details_simulateur=["Bienvenue sur le simulateur, l'interface peut paraître lourd...",
                    "mais pas d'inquiétude je vais tout expliquer.",
                    "Tout d'abord, vous pouvez choisir la nature de l'onde étudiée",
                    "en effet, chaque particule produit une onde en atteignant un écran.",
                    "En utilisant alors la physique quantique, et les probabilités,",
                    "vous pourrez obtenir sur votre ordinateur une visualisation !",
                    "Entrez les caractéristiques de la particule envoyée et le simulateur",
                    "simule 100 000 impacts (c'est beaucoup) et donne la figure d'interférence",
                    "associée. Ce n'est pas de la magie, mais de la physique et des formules !",
                    "Vous voyez à gauche, une figure rappelant le principe de l'expérience",
                    "des fentes de Young."]

Paranthèse_historique=["Lors des années 1900, la physique quantique n'est pas encore proprement définie.",
                       "L'étude des ondes étant émergente (débutant en 1895), les premiers modèles sont établis.",
                       "Le modèle du corps noir par Max Planck apparaît, or un problème est visible.",
                       "Comment expliquer la dépendance en fréquence pour l'énergie émise ?",
                       "Ce problème, interroge jusqu'à ce que Einstein découvre l'effet photoélectrique.",
                       "Enstein, en déduit l'existence de particules au vue de ses observations, dans son article :",
                       "Sur un point de vue heuristique concernant l'émission et la transformation de la lumière",
                       "(1905).",
                       "On parle alors de Quantas, et c'est la naissance de la théorie des Quantas.",
                       "La découverte d'Einstein aura des effets immenses, et sera considérée l'une",
                       "des plus importantes du siècle.",
                       "Suite à cela, plusieurs progrès seront faits : description des raies d'hydrogène"
                       "par Bohr, chimie quantique par London et Heitler et mécanique matricielle quantique",
                       "par Heisenberg et Born. On parvient alors, petit à petit à enfin décrire le monde",
                       "quantique et comprendre le réel.",
                       "On peut commencer à décrire l'électromagnétisme et Feynman, Dirac et d'autres scientifiques",
                       "se distinguent et aménent à des progrès réels qui serviront dans l'industrie des ordinateurs.",
                       "On pense notamment aux microprocesseurs, où hors l'informatique quantique, la loi de Moore",
                       "fut poussée à son maximum par l'emploi de ces mêmes propriétés quantiques.",
                       "Il en découle que plus qu'une science théorique, la physique quantique est au centre",
                       "de la construction du monde de demain.",
                       "Les images sont successivement (de haut en bas) celles de Planck, Heisenberg",
                       "et de Broglie."]

COULEUR_NOIRE = (0, 0, 0)
COULEUR_BLANCHE = (255, 255, 255)
COULEUR_ROUGE = (255, 0, 0)
COULEUR_ROUGE_SOMBRE = (185, 0, 0)
COULEUR_ROUGE_SOMBRE_2 = (125, 0, 0)
COULEUR_VERTE = (0, 255, 0)
COULEUR_VERTE_FORET = (0, 105, 0)
COULEUR_VERTE_FORET_SOMBRE = (0, 55, 0)
COULEUR_MAGENTA = (255, 0, 255)
COULEUR_MAGENTA_SOMBRE = (185, 0, 185)
COULEUR_MAGENTA_SOMBRE_2 = (125, 0, 125)
COULEUR_CYAN = (0, 255, 255)
COULEUR_CYAN_SOMBRE = (0, 185, 185)
COULEUR_CYAN_SOMBRE_2 = (0, 125, 125)
COULEUR_ORANGE = (255, 150, 0)
COULEUR_ORANGE_SOMBRE = (185, 100, 0)
COULEUR_ORANGE_SOMBRE_2 = (125, 55, 0)

font_titre = pygame.font.SysFont("times", 54, bold=True, italic=False)
font_texte = pygame.font.SysFont("times", 24, bold=True, italic=False)


Choix_utilisateur = 0
input_boxes = []
button_rect = pygame.Rect(350, 550, 180, 70)
mode_onde = "Plane"  
affichage_mode = "impacts"  
current_param_labels = parametres_par_mode["Plane"]  


rect_plane = pygame.Rect(1050, 250, 350, 50)
rect_spherique = pygame.Rect(1050, 320, 350, 50)
rect_conique = pygame.Rect(1050, 390, 350, 50)
rect_affichage = pygame.Rect(1050, 450, 350, 50)
rect_affichage_2 = pygame.Rect(1055, 455, 340, 40)
rect_affichage_3 = pygame.Rect(1060, 460, 330, 30)
def creer_input_boxes():
    global input_boxes
    input_boxes = []
    y_start = 250
    for i, label in enumerate(current_param_labels):
        box_x = 850
        box_y = y_start + 30 * i
        input_boxes.append(InputBox(box_x, box_y, 150, 30, font_texte, ""))


creer_input_boxes()


fin = 0
while fin == 0:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            fin = 1

        
        if event.type == pygame.KEYDOWN:
            active_exists = any(box.active for box in input_boxes)
            if not active_exists:
                if event.key == pygame.K_DOWN:
                    Choix_utilisateur += 1
                elif event.key == pygame.K_UP:
                    Choix_utilisateur -= 1

        
        if Choix_utilisateur == 0 and event.type == pygame.MOUSEBUTTONDOWN:
            if rect_plane.collidepoint(event.pos):
                mode_onde = "Plane"
                current_param_labels = parametres_par_mode["Plane"]
                creer_input_boxes()
                print("Mode : Ondes Planes")
            elif rect_spherique.collidepoint(event.pos):
                mode_onde = "Sphérique"
                current_param_labels = parametres_par_mode["Sphérique"]
                creer_input_boxes()
                print("Mode : Ondes Sphériques")
            elif rect_conique.collidepoint(event.pos):
                mode_onde = "Conique"
                current_param_labels = parametres_par_mode["Conique"]
                creer_input_boxes()
                print("Mode : Rayonnement Cherenkov")
            elif rect_affichage.collidepoint(event.pos):
                affichage_mode = "lisse" if affichage_mode == "impacts" else "impacts"
                print("Affichage :", affichage_mode)

        
        for box in input_boxes:
            box.handle_event(event)

        
        if Choix_utilisateur == 0 and event.type == pygame.MOUSEBUTTONDOWN and button_rect.collidepoint(event.pos):
            try:
                if mode_onde == "Plane" or mode_onde == "Sphérique":
                    
                    masse = float(input_boxes[0].text) if input_boxes[0].text else 9.11e-31
                    z = float(input_boxes[1].text) if input_boxes[1].text else 1.0
                    amplitude = float(input_boxes[2].text) if input_boxes[2].text else 1.0
                    N = int(input_boxes[3].text) if input_boxes[3].text else 1
                    x_vals = parse_float_list(input_boxes[4].text) or [0.1]
                    y_vals = parse_float_list(input_boxes[5].text) or [1.0]
                    ecart = float(input_boxes[6].text) if input_boxes[6].text else 0.1
                    energie = float(input_boxes[7].text) if input_boxes[7].text else 1e-19
                    if mode_onde == "Sphérique":
                        D_source = float(input_boxes[8].text) if input_boxes[8].text else 0.5
                    else:
                        D_source = None

                    
                    if len(x_vals) == 1:
                        x_vals = x_vals * N
                    if len(y_vals) == 1:
                        y_vals = y_vals * N
                    if len(x_vals) != N or len(y_vals) != N:
                        print("Erreur : dimensions incorrectes")
                        continue

                    
                    p = np.sqrt(2 * masse * energie)
                    lambda_ = h / p
                    
                    x_vals_m = [x * 0.01 for x in x_vals]
                    y_vals_m = [y * 0.01 for y in y_vals]
                    d_m = ecart * 0.01

                    
                    nx, ny = 800, 600
                    largeur_ecran = 0.02
                    x = np.linspace(-largeur_ecran/2, largeur_ecran/2, nx)
                    y = np.linspace(-largeur_ecran/2, largeur_ecran/2, ny)
                    X, Y = np.meshgrid(x, y)

                    if mode_onde == "Sphérique":
                        if N == 0:
                            I = calcul_source_seule_plane(X, Y, z, amplitude, lambda_)
                        else:
                            
                            nx_sph, ny_sph = 200, 200
                            x_sph = np.linspace(-largeur_ecran/2, largeur_ecran/2, nx_sph)
                            y_sph = np.linspace(-largeur_ecran/2, largeur_ecran/2, ny_sph)
                            X_sph, Y_sph = np.meshgrid(x_sph, y_sph)
                            I = calcul_spherique(X_sph, Y_sph, z, D_source, x_vals_m[0], y_vals_m[0], d_m, N, lambda_, amplitude)
                    else:  
                        if N == 0:
                            I = calcul_source_seule_plane(X, Y, z, amplitude, lambda_)
                        elif N == 1:
                            I = calcul_fente_unique_plane(X, Y, z, x_vals_m[0], y_vals_m[0], lambda_, amplitude)
                        elif N == 2:
                            I = calcul_double_fente_plane(X, Y, z, x_vals_m[0], y_vals_m[0], d_m, lambda_, amplitude)
                        else:
                            I = calcul_reseau_plane(X, Y, z, x_vals_m[0], y_vals_m[0], d_m, N, lambda_, amplitude)

                elif mode_onde == "Conique":
                    
                    beta = float(input_boxes[0].text) if input_boxes[0].text else 0.8
                    n_indice = float(input_boxes[1].text) if input_boxes[1].text else 1.33
                    L = float(input_boxes[2].text) if input_boxes[2].text else 1.0

                    nx, ny = 800, 600
                    largeur_ecran = 0.02
                    I = calcul_cherenkov(beta, L, n_indice, nx, ny, largeur_ecran)

                else:
                    print("Mode inconnu")
                    continue

                
                P = I / np.sum(I)

                
                N_part = 100000
                p_flat = P.ravel()
                indices = np.random.choice(p_flat.size, size=N_part, p=p_flat)
                i_indices, j_indices = np.unravel_index(indices, P.shape)
                image_impacts = np.zeros_like(P, dtype=int)
                np.add.at(image_impacts, (i_indices, j_indices), 1)

                if affichage_mode == "lisse":
                    image_array = P
                else:
                    image_array = image_impacts.astype(float)

                result_surface = intensite_vers_surface(image_array, use_colormap=True)
                print("Simulation terminée, ouverture de l'image...")
                sauvegarder_et_ouvrir(result_surface)

            except Exception as e:
                print(f"Erreur : {e}")

    
    if Choix_utilisateur >= len(Liste_options):
        Choix_utilisateur = 0
    elif Choix_utilisateur < 0:
        Choix_utilisateur = len(Liste_options) - 1

    
    fenetre.fill(COULEUR_BLANCHE)

    
    pygame.draw.rect(fenetre, COULEUR_VERTE, (0, 1, 300, 897), 2)
    pygame.draw.rect(fenetre, COULEUR_VERTE_FORET, (5, 5, 290, 889), 2)
    pygame.draw.rect(fenetre, COULEUR_VERTE_FORET_SOMBRE, (3, 3, 295, 892), 2)
    pygame.draw.rect(fenetre, COULEUR_ROUGE, (300, 0, 1200, 200), 2)
    pygame.draw.rect(fenetre, COULEUR_ROUGE_SOMBRE, (305, 5, 1190, 190), 2)
    pygame.draw.rect(fenetre, COULEUR_ROUGE_SOMBRE_2, (310, 10, 1180, 180), 2)

    
    text_titre = font_titre.render(Liste_options[Choix_utilisateur], 1, COULEUR_NOIRE)
    fenetre.blit(text_titre, (620, 70))

    
    for i, option in enumerate(Liste_options):
        text_option = font_texte.render(option, 1, COULEUR_NOIRE)
        fenetre.blit(text_option, (20, 10 + 30 * i))

    
    pygame.draw.rect(fenetre, COULEUR_ROUGE, (10, 10 + 30 * Choix_utilisateur, 280, 30), 2)
    pygame.draw.rect(fenetre, COULEUR_ROUGE_SOMBRE, (12, 12 + 30 * Choix_utilisateur, 276, 26), 2)
    pygame.draw.rect(fenetre, COULEUR_ROUGE_SOMBRE_2, (14, 14 + 30 * Choix_utilisateur, 272, 22), 2)

    
    if Choix_utilisateur == 2:  
        for i, ligne in enumerate(Description_Auteur):
            text_ligne = font_texte.render(ligne, 1, COULEUR_NOIRE)
            fenetre.blit(text_ligne, (550, 250 + 30 * i))
        image2 = pygame.transform.scale(image, (230, 230))
        fenetre.blit(image2, (305, 255))
        image_2 = pygame.transform.scale(image_2, (530, 130))
        image_3 = pygame.transform.scale(image_3, (280, 200))
        image_4 = pygame.transform.scale(image_4, (230, 230))
        fenetre.blit(image2, (305, 255))
        fenetre.blit(image_2, (320, 725))
        fenetre.blit(image_3, (875, 685))
        fenetre.blit(image_4, (1205, 655))

    elif Choix_utilisateur == 4:  
        for i, ligne in enumerate(Onglet_Quitter):
            text_ligne = font_texte.render(ligne, 1, COULEUR_NOIRE)
            fenetre.blit(text_ligne, (550, 250 + 30 * i))
        image2 = pygame.transform.scale(image, (230, 230))
        fenetre.blit(image2, (305, 255))
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
            pygame.quit()
        
    elif Choix_utilisateur == 1 :
        for i, ligne in enumerate(Paranthèse_historique):
            text_ligne = font_texte.render(ligne, 1, COULEUR_NOIRE)
            fenetre.blit(text_ligne, (550, 200 + 30 * i))
        Planck = pygame.transform.scale(Planck, (230, 230))
        fenetre.blit(Planck, (305, 205))
        Heisenberg = pygame.transform.scale(Heisenberg, (230, 230))
        fenetre.blit(Heisenberg, (305, 405))
        Broglie = pygame.transform.scale(Broglie, (230, 230))
        fenetre.blit(Broglie, (305, 605))
    elif Choix_utilisateur == 3:  
        for i, ligne in enumerate(Formules_Mathematiques):
            text_ligne = font_texte.render(ligne, 1, COULEUR_NOIRE)
            fenetre.blit(text_ligne, (550, 200 + 30 * i))
        image2 = pygame.transform.scale(image, (230, 200))
        fenetre.blit(image2, (305, 205))
        pdf = pygame.transform.scale(pdf, (230, 300))
        fenetre.blit(pdf, (305, 415))

    elif Choix_utilisateur == 0:  
        
        for i, label in enumerate(current_param_labels):
            text_label = font_texte.render(label, 1, COULEUR_NOIRE)
            fenetre.blit(text_label, (350, 250 + 30 * i))
            if i < len(input_boxes):
                input_boxes[i].draw(fenetre)
        for i, ligne in enumerate(Details_simulateur):
            text_ligne = font_texte.render(ligne, 1, COULEUR_NOIRE)
            fenetre.blit(text_ligne, (650, 550 + 30 * i))
        
        pygame.draw.rect(fenetre, COULEUR_ROUGE, button_rect)
        pygame.draw.rect(fenetre, COULEUR_ROUGE_SOMBRE, (355, 555, 170, 60))
        pygame.draw.rect(fenetre, COULEUR_ROUGE_SOMBRE_2, (360, 560, 160, 50))
        text_button = font_texte.render("Lancer", True, COULEUR_NOIRE)
        fenetre.blit(text_button, (button_rect.x + 50, button_rect.y + 20))
        young = pygame.transform.scale(young, (420, 270))
        fenetre.blit(young, (269, 615))
        
        pygame.draw.rect(fenetre, COULEUR_ROUGE, rect_plane)
        pygame.draw.rect(fenetre, COULEUR_ROUGE_SOMBRE, (1055, 255, 340, 40))
        pygame.draw.rect(fenetre, COULEUR_ROUGE_SOMBRE_2, (1060, 260, 330, 30))
        pygame.draw.rect(fenetre, COULEUR_MAGENTA, rect_spherique)
        pygame.draw.rect(fenetre, COULEUR_MAGENTA_SOMBRE, (1055, 325, 340, 40))
        pygame.draw.rect(fenetre, COULEUR_MAGENTA_SOMBRE_2, (1060, 330, 330, 30))
        pygame.draw.rect(fenetre, COULEUR_CYAN, rect_conique)
        pygame.draw.rect(fenetre, COULEUR_CYAN_SOMBRE, (1055, 395, 340, 40))
        pygame.draw.rect(fenetre, COULEUR_CYAN_SOMBRE_2, (1060, 400, 330, 30))

        text_plane = font_texte.render("Ondes Planes", True, COULEUR_NOIRE)
        fenetre.blit(text_plane, (1100, 260))
        text_spherique = font_texte.render("Ondes Sphériques", True, COULEUR_NOIRE)
        fenetre.blit(text_spherique, (1100, 330))
        text_conique = font_texte.render("Rayonnement Cherenkov", True, COULEUR_NOIRE)
        fenetre.blit(text_conique, (1100, 400))

        
        pygame.draw.rect(fenetre, COULEUR_ORANGE, rect_affichage)
        pygame.draw.rect(fenetre, COULEUR_ORANGE_SOMBRE, rect_affichage_2)
        pygame.draw.rect(fenetre, COULEUR_ORANGE_SOMBRE_2, rect_affichage_3)
        text_aff = font_texte.render(f"Affichage : {affichage_mode}.", True, COULEUR_NOIRE)
        fenetre.blit(text_aff, (1100, 460))
        
        if mode_onde == "Plane":
            pygame.draw.rect(fenetre, COULEUR_NOIRE, rect_plane, 5)
        elif mode_onde == "Sphérique":
            pygame.draw.rect(fenetre, COULEUR_NOIRE, rect_spherique, 5)
        elif mode_onde == "Conique":
            pygame.draw.rect(fenetre, COULEUR_NOIRE, rect_conique, 5)
    text_aff = font_texte.render("Simulateur réalisé par", True, COULEUR_NOIRE)
    fenetre.blit(text_aff, (30, 200))
    text_aff = font_texte.render("DAOUADI Zine-Eddine", True, COULEUR_NOIRE)
    fenetre.blit(text_aff, (25, 230))
    text_aff = font_texte.render("Lycée Janson-De-Sailly", True, COULEUR_NOIRE)
    fenetre.blit(text_aff, (25, 300))
    Janson = pygame.transform.scale(Janson, (200, 200))
    fenetre.blit(Janson, (50, 350))
    text_aff = font_texte.render("Université Paris-Saclay.", True, COULEUR_NOIRE)
    fenetre.blit(text_aff, (25, 560))
    Saclay = pygame.transform.scale(Saclay, (230, 200))
    fenetre.blit(Saclay, (30, 625))
        
    pygame.display.flip()

pygame.quit()
