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
- Pour exécuter le projet, saisissez la commande `python main.py` dans votre terminal à partir du répertoire `interface/`.
- Le modèle préentraîné est disponible au téléchargement via ce lien : [E-30-35-98,83.keras](https://drive.google.com/file/d/1lv2bGIFW4VsZqR-HX_lc6dKgVCMoSaL1/view?usp=drive_link).
- Pour atteindre une précision finale de 98.83 %, un travail d’optimisation des hyperparamètres a été nécessaire. L’objectif était de trouver la meilleure combinaison possible pour maximiser la performance du modèle lors de la phase d’entraînement.
- Plusieurs expérimentations ont donc été menées en variant notamment l’architecture de base (MobileNetV2, ResNet50, EfficientNetB3, ConvNeXt), le taux de dropout, le nombre de blocs à décongeler dans le modèle pré-entraîné ainsi que le nombre d’epochs d'entraînement.
- Tous les modèles ont été exécutés sur Google Colab avec une accélération GPU T4 et la précision finale a été atteinte après un entraînement d’une durée de **3 minutes et 43 secondes**. Voici un tableau récapitulatif des différents tests effectués :
  
| Architecture | Droupout (%) | Blocs dégelés | Epochs | Accuracy (%) |
|:------:|:------:|:------:|:------:| :------:|
| MobileNetV2 | 20 | 2 | 10 | 45,18 |
| MobileNetV2 | 55 | 1 | 10 | 56,87 |
| MobileNetV2 | 0  | 0 | 10 | 56,98 |
| MobileNetV2 | 40 | 1 | 10 | 57,31 |
| ResNet50 | 45 | 35 | 20 | 93,42 |
| ResNet50 | 40 | 1  | 20 | 95,03 |
| ResNet50 | 35 | 2  | 20 | 95,32 |
| ResNet50 | 50 | 35 | 20 | 95,61 |
| ResNet50 | 40 | 32 | 20 | 95,61 |
| ResNet50 | 35 | 10 | 20 | 96,05 |
| ResNet50 | 42 | 33 | 20 | 96,05 |
| ResNet50 | 45 | 30 | 20 | 96,78 |
| EfficientNetB3 | 30 | 0  | 20 | 96,64 |
| EfficientNetB3 | 28 | 38 | 20 | 97,85 |
| EfficientNetB3 | 40 | 30 | 20 | 97,81 |
| EfficientNetB3 | 20 | 1  | 20 | 97,95 |
| EfficientNetB3 | 45 | 19 | 20 | 97,95 |
| EfficientNetB3 | 45 | 35 | 20 | 97,95 |
| EfficientNetB3 | 50 | 35 | 20 | 97,95 |
| EfficientNetB3 | 45 | 21 | 20 | 98,10 |
| EfficientNetB3 | 30 | 23 | 20 | 98,10 |
| EfficientNetB3 | 32 | 36 | 20 | 98,10 |
| EfficientNetB3 | 45 | 30 | 20 | 98,25 |
| EfficientNetB3 | 35 | 30 | 20 | 98,39 |
| EfficientNetB3 | 35 | 35 | 20 | 98,39 |
| EfficientNetB3 | 35 | 25 | 20 | 98,54 |
| EfficientNetB3 | 45 | 16 | 20 | 98,68 |
| EfficientNetB3 | 33 | 37 | 20 | 98,68 |
| EfficientNetB3 | 30 | 35 | 20 | 98,83 |
| ConvNeXt | 50 | 40 | 20 | 97,95 |
| ConvNeXt | 35 | 25 | 20 | 98,10 |
| ConvNeXt | 40 | 25 | 20 | 98,39 |
| ConvNeXt | 30 | 15 | 20 | 98,54 |
| ConvNeXt | 50 | 15 | 20 | 98,54 |
| ConvNeXt | 30 | 35 | 20 | 98,58 |
| ConvNeXt | 30 | 30 | 20 | 98,68 |
| ConvNeXt | 30 | 25 | 20 | 98,68 |
| ConvNeXt | 40 | 15 | 20 | 98,68 |
| ConvNeXt | 35 | 15 | 20 | 98,68 |
| ConvNeXt | 30 | 40 | 20 | 98,83 |
| ConvNeXt | 50 | 30 | 20 | 98,83 |
| ConvNeXt | 50 | 35 | 20 | 98,83 |
| ConvNeXt | 50 | 20 | 20 | 98,83 |