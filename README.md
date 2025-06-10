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
- Pour atteindre une précision finale de 98.83 %, un travail d’optimisation des hyperparamètres a été nécessaire. L’objectif était de trouver la meilleure combinaison possible pour maximiser la performance du modèle lors de la phase d’entraînement.
- Plusieurs expérimentations ont été menées en variant notamment l’architecture de base (MobileNetV2, ResNet50, EfficientNetB3, ConvNeXt), le taux de dropout, le nombre de blocs à décongeler dans le modèle pré-entraîné ainsi que le nombre d’époques d'entraînement. De plus, tous les modèles ont été exécutés sur Google Colab, avec accélération GPU. Voici un récapitulatif des différents tests effectués :
| Architecture | Droupout (%) | Blocs dégelés | Epochs | Accuracy (%) |
|:------:|:------:|:------:|:------:| :------:|
| MobileNetV2  | 55| 1|10 |56,87 |
| MobileNetV2 | 0|0 |10 | 56,98|
| MobileNetV2 | 40| 1|10 |57,31 |
| MobileNetV2 |20 |2 | 10| 45,18 |
| MobileNetV2 |40 | 1| 10|57,31 |
| MobileNetV2 | 55|1 | 10|56,87 |
| ResNet50 | 35 |2 |20 |95,32 |
| ResNet50 |35 | 10|20 |96,05 |
| ResNet50 |40 | 1| 20| 95,03|
|  | | | | |
|  | | | | |
|  | | | | |
|  | | | | |
|  | | | | |
|  | | | | |
|  | | | | |
