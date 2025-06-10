# Bird-Recognition
Modèle de classification d’images basé sur EfficientNetB3 permettant l’identification de 20 espèces d’oiseaux à partir de photos avec lecture automatique du chant correspondant.

# Fonctionnalités 
- Classification de 20 espèces d’oiseaux via un modèle d’apprentissage profond basé sur EfficientNetB3 avec une précision de 98.83 %.
- Lecture automatique du chant de l’oiseau identifié.

# Structure du projet
- images/ : Contient les 3408 photos d’oiseaux réparties en 20 classes distinctes.
- interface/ : Contient l’interface graphique de l’application.
- model/ : Contient le notebook Python retraçant l’entraînement et la création du modèle.
- sons/ : Contient les chants des 20 espèces d’oiseaux.

# Prérequis
- Python version 3.x
- Les packages : tensorflow, matplotlib, tkinter, Pillow, numpy, os, playsound & glob.

# Note
-
| Colonne 1 | Colonne 2 | Colonne 3 |
|-----------|-----------|-----------|
| Ligne 1   | Donnée A  | Donnée B  |
| Ligne 2   | Donnée C  | Donnée D  |