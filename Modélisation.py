# -*- coding: utf-8 -*-
"""
Created on 2 Dec 2020

@author: chatoux
"""

# Nombre de coins intérieurs par ligne et colonne d'échiquier
# Pour trouver ca, compter le nombre d'intercection noir/blanc sur largeur/longueur de l'échiquier -> ici 4 et 6
largeur = 4
longueur = 6
# largeur des carreaux du damier en mm
largeurCarreaux = 40


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy import ndimage
import glob

# Exercice 1
def CameraCalibration():
    # critères d'arret --> epsilon = 0.001 , max iteration = 30
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((largeur * longueur, 3), np.float32)
    objp[:, :2] = np.mgrid[0:largeur, 0:longueur].T.reshape(-1, 2)
    # Mise à l'echelle
    objp[:, :2] *= largeurCarreaux

    # tableaux pour stocker les points pour toutes les images de l'échiquier
    objpoints = []  # points en 3D
    imgpoints = []  # points en 2D dans le plan image

    # forme une seule variable pour toutes les images, cela nous permettra de toutes les parcourir plus facilement
    # liste de nom sous forme de chaine
    images = glob.glob('./Images/chess/P30/*.jpg')

    # pour chaque image (voir ligne juste au dessus pour images)
    for i, fname in enumerate(images):
        # on lit l'image que l'on traitera lors de cette itération avec cv.imread()
        img = cv.imread(fname)
        # on brouille l'image et la sous-échantillonne avec cv.pyrDown()
        img = cv.pyrDown(img)
        # Convertit l'image d'un espace colorimétrique à un autre avec cv.cvtColor
        # arguments: InputArray -> img: l'image traitée lors de cette itération
        #            Code       -> cv.COLOR_BGR2GRAY: code pour la conversion de RVB/BGR en niveaux de gris
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Trouver les coins de l'échiquier
        # arguments : InputArray  -> gray : Vue de l'échiquier source. Il doit s'agir d'une image en niveaux de gris ou en couleurs 8 bits.
        #             Size        -> (largeur, longueur) : Nombre de coins intérieurs par ligne et colonne d'échiquier
        #             OutputArray -> None : ici on récupère direct les coins detecté dans la variable corners
        #             ret prend la valeur true si les coins sont trouvés et false sinon
        ret, corners = cv.findChessboardCorners(gray, (largeur, longueur), None)

        # on affiche le résultat de findChessboardCorners
        print("Résulat détection des coins pour l'image n°", i,": ", ret)

        # Si suffisamment de coins ont été trouvé
        if ret == True:
            # Ajoute à la liste des points des objets une copie des points des objets préparés lignes 26/27
            objpoints.append(objp)
            # Affine les emplacements des coins
            # arguments : InputArray -> gray: image source, en 8 bits
            #             Corners    -> corners: Coordonnées initiales des coins d'entrée
            #             WinSize    -> (11, 11): La moitié de la longueur du côté de la fenêtre de recherche. Par exemple, si winSize = Size (5,5), alors un ( 5 ∗ 2 + 1 ) × ( 5 ∗ 2 + 1 ) = 11 × 11
            #             ZeroZone   -> (-1, -1): La moitié de la taille de la région morte au milieu de la zone de recherche sur laquelle la sommation dans la formule ci-dessous n'est pas effectuée.
            #                                   Il est parfois utilisé pour éviter d'éventuelles singularités de la matrice d'autocorrélation.
            #                                   La valeur de (-1, -1) indique qu'une telle taille n'existe pas.
            #             Criteria   -> criteria: Critères de fin du processus itératif de raffinement d'angle.
            #                                   Autrement dit, le processus de raffinement de la position d'angle s'arrête soit après les itérations de critère.maxCount, soit lorsque la position d'angle se déplace de moins que critère.epsilon sur une itération.
            #             Sortie     -> corners2: coordonnées raffinées des coins
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # on ajoute à la liste des points des images, la position des coins trouvés
            imgpoints.append(corners)
            # Dessine sur l'échiqiuer les coins qui ont été détectés. Soit comme des cercles rouges si il n'a pas été trouvé, soit comme des points colorés reliés par des lignes si il a été trouvé
            # arguments: InputOutputArray -> img: l'image de l'itération cours sans la conversion en niveaux de gris, sur laquelle on va dessiner
            #            PatternSize      -> (largeur, longueur): Nombre de coins intérieurs par ligne et colonne d'échiquier
            #            Corner           -> corners2: coordonnées raffinées des coins
            #            PatternWasFound  -> ret: retour de findChessboardCorners, ici forcement true grâce au "if" ligne 61, true = coins detectés
            cv.drawChessboardCorners(img, (largeur, longueur), corners2, ret)
            # affichage de l'image sur laquelle on vient de dessiner dans une fenetre
            cv.namedWindow('img', 0)
            cv.imshow('img', img)
            # permet de faire une pause
            #    waitKey attend qu'on presse une touche. Ici la fonction attend pendant seulement 0.5s comme on lui a indiqué passant un argument (500 ms)
            cv.waitKey(500)
    # on ferme toutes les fenetres
    cv.destroyAllWindows()

    # Recherche les paramètres intrinsèques et extrinsèques de la caméra
    # arguments: InputArrayOfArrays -> objpoints: tableau des points en 3D connus
    #            InputArrayOfArrays -> imgpoints: tableau des coordonnées 2D des coins connus
    #            Size               -> gray.shape[::-1]: taille de l'image (ici 1368 x 1824)
    #            Flags              -> None
    #            Criteria           -> None
    # sorties: boolean     -> ret: valeur de retour, true ou false si c'est une résussite ou un échec
    #          OutputArray -> mtx: Matrice intrinsèque de caméra (Matrice de calibrage 3x3)
    #          OutputArray -> dist: Coefficients de distorsion de l'objectif
    #          OutputArray -> rvecs: vecteurs de rotation estimé pour chaque vue
    #          OutputArray -> tvecs: vecteurs de traduction estimés pour chaque vue
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("\nRésultats de cv.calibrateCamera:")
    print('camraMatrix\n', mtx, "\n")
    print('dist\n', dist, "\n"  )

    # on selectionne une nouvelle image (imread) et on la sous-echantillonne (pyDown)
    img = cv.pyrDown(cv.imread('./Images/chess/P30/IMG_20201206_093855.jpg'))
    # Dimension de l'image -> h = hauteur, w = largeur
    h, w = img.shape[:2]
    # Calcule d'une nouvelle matrice de calibrage en optimisant celle précedement trouvée
    # Ici ceci nous permet de corriger la distorsion induite par l'appareil photo
    # arguments: InputArray           -> mtx: matrice de calibrage d'entrée
    #            InputArray           -> dist: tableau des coefficients de distorsion
    #            ImageSize            -> (w, h): taille de l'image d'origine
    #            Alpha                -> 1: Paramètre de mise à l'échelle libre entre 0 (lorsque tous les pixels de l'image non déformée sont valides) et 1 (lorsque tous les pixels de l'image source sont conservés dans l'image non déformée).
    #            newImgSize           -> (w, h): taille de l'image après rectification
    #            centerPrincipalPoint -> None: Indicateur facultatif qui indique si, dans la nouvelle matrice intrinsèque de la caméra, le point principal doit être au centre de l'image ou non.
    #                                    Par défaut, le point principal est choisi pour adapter au mieux un sous-ensemble de l'image source (déterminé par alpha) à l'image corrigée.
    # sortie: newcameramtx: Nouvelle matrice de calibrage obtnue apreès la rectification
    #         roi: Rectangle de sortie facultatif qui décrit la région de tous les bons pixels dans l'image non déformée.
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # on affiche la nouvelle matrice
    print('newcameramtx\n', newcameramtx)

    # On souhaite corriger la distortion
    #  Calcule la carte de transformation
    # arguments: InputArray -> mtx: matrice de calibrage de la caméra avant rectification
    #            InputArray -> dist: tableau des coefficients de distorsion
    #            InputArray -> None: Transformation de rectification optionnelle dans l'espace objet (matrice 3x3)
    #            InputArray -> newcameramtx: matrice de calibrage apres rectification
    #            ImageSize  -> (w, h): taille de l'image non déformée
    #            M1Type     -> 5: type de la première carte de sortie
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    # utilisation d'une fonction de remappage pour cooriger la distorion
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # Enfin on recadre l'image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    # On affiche le résultat
    cv.namedWindow('img', 0)
    cv.imshow('img', dst)
    # 0 en paramètre de waitKey signifit qu'il attendra pour toujours tant qu'on ne presse pas de touche
    cv.waitKey(0)
    # on stocke le résultat en créant une nouvelle image
    cv.imwrite('./Images/Resultats/calibresultM.png', dst)

    # variable pour l'erreur de re-projection
    # Donne une bonne estimation de l'exactitude des paramètres trouvés. Plus l'erreur de re-projection est proche de zéro, plus les paramètres que nous avons trouvés sont précis
    mean_error = 0
    for i in range(len(objpoints)):
        #  nous devons d'abord transformer l'objet point en point image en utilisant cv.projectPoints ()
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        # Ensuite, nous pouvons calculer la norme absolue entre ce que nous avons obtenu avec notre transformation et l'algorithme de recherche de coin
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    # Pour trouver l'erreur moyenne, nous calculons la moyenne arithmétique des erreurs calculées pour toutes les images d'étalonnage
    # et on affiche
    print("\ntotal error: {}".format(mean_error / len(objpoints)))

    # on retourne la matrice de calibrage de notre caméra
    return newcameramtx

"""
# exercice 4
def DepthMapfromStereoImages():
    ############ to adapt ##########################
    imgL = cv.pyrDown(cv.imread('Images/aloeL.jpg'))
    imgR = cv.pyrDown(cv.imread('Images/aloeR.jpg'))
    #################################################
    # 
    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=16,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=16,
                                  uniquenessRatio=10,
                                  speckleWindowSize=100,
                                  speckleRange=32)
    # 
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    # 
    plt.figure('3D')
    plt.imshow((disparity - min_disp) / num_disp, 'gray')
    plt.colorbar()
    plt.show()


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 2)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# exercice 2
def StereoCalibrate(Cameramtx):
    ############ to adapt ##########################
    img1 = cv.pyrDown(cv.imread('Images/leftT2.jpg', 0))
    img2 = cv.pyrDown(cv.imread('Images/rightT2.jpg', 0))
    #################################################
    # opencv 4.5
    sift = cv.SIFT_create()
    # opencv 3.4
    #sift = cv.xfeatures2d.SIFT_create()
    # 
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # 
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # 
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    plt.imshow(img3)
    plt.show()

    # 
    E, maskE = cv.findEssentialMat(pts1, pts2, Cameramtx, method=cv.FM_LMEDS)
    print('E\n', E)
    # 
    retval, R, t, maskP = cv.recoverPose(E, pts1, pts2, Cameramtx, maskE)
    print('R\n', R)
    print('t\n', t)

    # Calcul de la matrice fondamentale à partir de la matrice essentielle

    #
    F, maskF = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    print('F\n', F)

    return pts1, pts2, F, maskF, FT, maskE

# Exercice 3
def EpipolarGeometry(pts1, pts2, F, maskF, FT, maskE):
    ############ to adapt ##########################
    img1 = cv.pyrDown(cv.imread('Images/leftT2.jpg', 0))
    img2 = cv.pyrDown(cv.imread('Images/rightT2.jpg', 0))
    #################################################
    r, c = img1.shape

    # 
    pts1F = pts1[maskF.ravel() == 1]
    pts2F = pts2[maskF.ravel() == 1]

    # 
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1F, pts2F)
    # 
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    plt.figure('Fright')
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img6)
    plt.figure('Fleft')
    plt.subplot(121), plt.imshow(img4)
    plt.subplot(122), plt.imshow(img3)

    # 
    pts1 = pts1[maskE.ravel() == 1]
    pts2 = pts2[maskE.ravel() == 1]
    # 
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, FT)
    lines1 = lines1.reshape(-1, 3)
    img5T, img6T = drawlines(img1, img2, lines1, pts1, pts2)
    plt.figure('FTright')
    plt.subplot(121), plt.imshow(img5T)
    plt.subplot(122), plt.imshow(img6T)
    # 
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, FT)
    lines2 = lines2.reshape(-1, 3)
    img3T, img4T = drawlines(img2, img1, lines2, pts2, pts1)
    plt.figure('FTleft')
    plt.subplot(121), plt.imshow(img4T)
    plt.subplot(122), plt.imshow(img3T)
    plt.show()

    # 
    retval, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, (c, r))
    print(H1)
    print(H2)
    # 
    im_dst1 = cv.warpPerspective(img1, H1, (c, r))
    im_dst2 = cv.warpPerspective(img2, H2, (c, r))
    cv.namedWindow('left', 0)
    cv.imshow('left', im_dst1)
    cv.namedWindow('right', 0)
    cv.imshow('right', im_dst2)
    cv.waitKey(0)
"""


if __name__ == "__main__":
    # determine la matrice de calibrage de notre caméra grâce à un échantillon d'image
    cameraMatrix = CameraCalibration()

    """
    
    cameraMatrix = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])


    dist = [[0, 0, 0, 0, 0]]

    # Exercice 2: Calibrage de la stéréovision
    pts1, pts2, F, maskF, FT, maskE = StereoCalibrate(cameraMatrix)

    # Exercice 3: Reconstruction des lignes épipolaires
    EpipolarGeometry(pts1, pts2, F, maskF, FT, maskE)

    # exercice 4 bonus
    DepthMapfromStereoImages()
    
    """