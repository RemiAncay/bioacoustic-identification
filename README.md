# Rémi Ançay - Travail de Bachelor

Ce git contient les scripts et notebooks nécessaires pour reproduire les expériences réalisées dans le cadre de mon travail de Bachelor à la HEIG-VD.

Il ne contient pas les données d'entrainement et de test ainsi que les poids des modèles entraînés. Ces données sont trop lourdes pour être hébergées sur GitHub.

## Identification des loups gris par leurs hurlements

Ce projet s’inscrit dans le domaine de la bioacoustique, avec pour objectif de développer des méthodes d’identification automatique à partir d’enregistrements sonores. Il s’appuie sur des techniques de machine learning pour reconnaître, d’une part, les espèces animales à partir de leurs vocalisations, et d’autre part, les individus au sein d’une même espèce en analysant leurs signatures vocales spécifiques.

Deux types de modèles ont été étudiés et comparés : BirdNET, un système basé sur des réseaux de neurones convolutionnels, initialement conçu pour la reconnaissance d’oiseaux, mais qui a été, ici, réentraîné pour nos tâches ; et AST (Audio Spectrogram Transformer), un modèle plus récent utilisant une architecture de type Transformer, adaptée au traitement de spectrogrammes audio. Ces deux approches ont été testées, comparées, et entraînées avec des jeux de données adaptés aux deux tâches visées.
Le projet a nécessité plusieurs étapes, incluant la recherche de données pertinentes, le prétraitement des enregistrements, la structuration des jeux de données, l’entraînement des modèles et l’analyse des résultats. Ces expériences ont permis de valider la faisabilité de l’identification automatique, même dans des cas où les différences sonores sont très subtiles.

Les résultats obtenus sont excellents, et confirment l’efficacité des modèles testés pour des tâches complexes d’identification bioacoustique. Ce travail montre le potentiel de ces outils pour des applications concrètes en écologie, en conservation des espèces ou en suivi de la biodiversité, tout en soulignant les défis liés à la qualité des données et à la robustesse des modèles.


## Architecture du projet
Le projet est organisé en plusieurs répertoires :
- `Preprocessing/` : Contient un script permettant de faire plusieurs opérations de prétraitement sur les données audio, ainsi qu'un notebook permettant de télécharger et de sauvegarder de manière simplifiée un dataset provenant de HuggingFace.
- `BirdNET/` : Contient deux notebooks permettant d'évaluer le modèle BirdNET sur des données de test, ainsi que de visualiser et calculer les métriques de performance de prédiction d'un modèle BirdNET.
- `AST/` : Contient deux notebooks permettant d'entrainer et de tester un modèle AST (Audio Spectrogram Transformer).
- `AudioClassificationGame/` : Contient un jeu de classification audio permettant de se rendre compte de la difficulté de la tâche de classification audio.

Le projet à été développé et testé avec Python 3.11.11.
Le fichier `requirements.txt` contient les dépendances nécessaires pour faire fonctionner les notebooks.
Pour les installer, il suffit d'exécuter la commande suivante dans le terminal :

```bash
pip install -r requirements.txt
```

## Contact
Pour toute question ou information complémentaire, vous pouvez me contacter par email à l'adresse suivante : remi.ancay@gmail.com
