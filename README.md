# Projet Machine Learning avec PySpark - Prédiction du Churn Télécom
### Description du modèle ML choisi
##### Objectif :
Le but du modèle est de prédire si un client d’un opérateur télécom va résilier son abonnement (churn)
ou non. Cette prédiction permet d’anticiper les départs clients et de mettre en place des actions ciblées
pour réduire le churn. 
##### Algorithme :
Nous avons choisi d’utiliser un classificateur Random Forest implémenté avec PySpark MLlib. Random
Forest est un ensemble d’arbres de décision qui réduit le surapprentissage et améliore la précision
globale. C’est un algorithme robuste et efficace pour les données tabulaires hétérogènes.
##### Métriques :
Pour évaluer la performance du modèle, nous utilisons :
Accuracy (précision globale)
### Description du dataset utilisé
Nom du fichier : Data4.csv
Taille : 7043 lignes , 21 colonnes
Format : CSV, avec une colonne cible Churn (valeurs binaires 0 ou 1)
Source : Dataset fictif / téléchargeable sur un référentiel pédagogique (à adapter selon source réelle)
Colonnes : Informations clients comme sexe, Client senior, contrat, Partenaire, Service téléphonique, etc. 
Certaines colonnes catégorielles doivent être encodées.
### Détails sur l’adaptation du modèle à PySpark
SparkSession: point d’entré de Spark.
StringIndexer: Transforme du texte ⟶ indice numerique.
EXEMPLE: Gender (Male/Female) ⟶ Gender_indexer (0/1)
VectorAssembler : assemble plusieurs colonne en une seule features.
RandomForestClassifier: Algorithme de classification binaire avancé.
MulticlassClassificationEvaluator: Evaluation du modéle (calculer les métriques d’accuracy).
