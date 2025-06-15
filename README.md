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

<div align="center">

<table>
  <thead>
    <tr>
      <th>Architecture</th>
      <th>Droupout (%)</th>
      <th>Blocs dégelés</th>
      <th>Epochs</th>
      <th>Accuracy (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>MobileNetV2</td><td>20</td><td>2</td><td>10</td><td>45,18</td></tr>
    <tr><td>MobileNetV2</td><td>55</td><td>1</td><td>10</td><td>56,87</td></tr>
    <tr><td>MobileNetV2</td><td>0</td><td>0</td><td>10</td><td>56,98</td></tr>
    <tr><td>MobileNetV2</td><td>40</td><td>1</td><td>10</td><td>57,31</td></tr>
    <tr><td>ResNet50</td><td>45</td><td>35</td><td>20</td><td>93,42</td></tr>
    <tr><td>ResNet50</td><td>40</td><td>1</td><td>20</td><td>95,03</td></tr>
    <tr><td>ResNet50</td><td>35</td><td>2</td><td>20</td><td>95,32</td></tr>
    <tr><td>ResNet50</td><td>50</td><td>35</td><td>20</td><td>95,61</td></tr>
    <tr><td>ResNet50</td><td>40</td><td>32</td><td>20</td><td>95,61</td></tr>
    <tr><td>ResNet50</td><td>35</td><td>10</td><td>20</td><td>96,05</td></tr>
    <tr><td>ResNet50</td><td>42</td><td>33</td><td>20</td><td>96,05</td></tr>
    <tr><td>ResNet50</td><td>45</td><td>30</td><td>20</td><td>96,78</td></tr>
    <tr><td>EfficientNetB3</td><td>30</td><td>0</td><td>20</td><td>96,64</td></tr>
    <tr><td>EfficientNetB3</td><td>28</td><td>38</td><td>20</td><td>97,85</td></tr>
    <tr><td>EfficientNetB3</td><td>40</td><td>30</td><td>20</td><td>97,81</td></tr>
    <tr><td>EfficientNetB3</td><td>20</td><td>1</td><td>20</td><td>97,95</td></tr>
    <tr><td>EfficientNetB3</td><td>45</td><td>19</td><td>20</td><td>97,95</td></tr>
    <tr><td>EfficientNetB3</td><td>45</td><td>35</td><td>20</td><td>97,95</td></tr>
    <tr><td>EfficientNetB3</td><td>50</td><td>35</td><td>20</td><td>97,95</td></tr>
    <tr><td>EfficientNetB3</td><td>45</td><td>21</td><td>20</td><td>98,10</td></tr>
    <tr><td>EfficientNetB3</td><td>30</td><td>23</td><td>20</td><td>98,10</td></tr>
    <tr><td>EfficientNetB3</td><td>32</td><td>36</td><td>20</td><td>98,10</td></tr>
    <tr><td>EfficientNetB3</td><td>45</td><td>30</td><td>20</td><td>98,25</td></tr>
    <tr><td>EfficientNetB3</td><td>35</td><td>30</td><td>20</td><td>98,39</td></tr>
    <tr><td>EfficientNetB3</td><td>35</td><td>35</td><td>20</td><td>98,39</td></tr>
    <tr><td>EfficientNetB3</td><td>35</td><td>25</td><td>20</td><td>98,54</td></tr>
    <tr><td>EfficientNetB3</td><td>45</td><td>16</td><td>20</td><td>98,68</td></tr>
    <tr><td>EfficientNetB3</td><td>33</td><td>37</td><td>20</td><td>98,68</td></tr>
    <tr><td>EfficientNetB3</td><td>30</td><td>35</td><td>20</td><td>98,83</td></tr>
    <tr><td>ConvNeXt</td><td>50</td><td>40</td><td>20</td><td>97,95</td></tr>
    <tr><td>ConvNeXt</td><td>35</td><td>25</td><td>20</td><td>98,10</td></tr>
    <tr><td>ConvNeXt</td><td>40</td><td>25</td><td>20</td><td>98,39</td></tr>
    <tr><td>ConvNeXt</td><td>30</td><td>15</td><td>20</td><td>98,54</td></tr>
    <tr><td>ConvNeXt</td><td>50</td><td>15</td><td>20</td><td>98,54</td></tr>
    <tr><td>ConvNeXt</td><td>30</td><td>35</td><td>20</td><td>98,58</td></tr>
    <tr><td>ConvNeXt</td><td>30</td><td>30</td><td>20</td><td>98,68</td></tr>
    <tr><td>ConvNeXt</td><td>30</td><td>25</td><td>20</td><td>98,68</td></tr>
    <tr><td>ConvNeXt</td><td>40</td><td>15</td><td>20</td><td>98,68</td></tr>
    <tr><td>ConvNeXt</td><td>35</td><td>15</td><td>20</td><td>98,68</td></tr>
    <tr><td>ConvNeXt</td><td>30</td><td>40</td><td>20</td><td>98,83</td></tr>
    <tr><td>ConvNeXt</td><td>50</td><td>30</td><td>20</td><td>98,83</td></tr>
    <tr><td>ConvNeXt</td><td>50</td><td>35</td><td>20</td><td>98,83</td></tr>
    <tr><td>ConvNeXt</td><td>50</td><td>20</td><td>20</td><td>98,83</td></tr>
  </tbody>
</table>

</div>