## Projet de détection de deepfake audio

Notre projet vise à mettre en place un pipeline complet permettant de détecter des deepfakes audio en utilisant des algorithmes de machine learning.
Pour ce faire on entraine notre modèle sur le jeu de données ASVspoof 2019. Ce data set contient deux types de fichiers audio, les PA (Adresses Physiques) et les LA (Adresses Logiques). Nous utiliserons les LA pour notre projet car ce sont des audios générés par IA. Les PA sont des relectures d'enregistrements déjà existant, donc pas utiles pour notre projet. On va utiliser deux sous-ensembles fournies avec le data set, train pour l'entrainement du modèle et dev pour la validation et l'évaluation.

---

## Présentation du dataset ASVspoof 2019-LA

Le dataset ASVspoof 2019-LA contient plusieurs fichiers audio .flac labélisés pouvant être des audios synthétiques générés par IA (spoof) ou bien de réels enregistrements audio (bonafide). 
Ses audios sont divisés en deux catégories, les données d'entrainement 'train' et les données d'évaluations. Elles sont labellisés dans les fichiers protocoles ASVspoof2019.LA.cm.train.trn.txt (entrainement) et ASVspoof2019.LA.cm.dev.trl.txt.
Le dataset fourni également un troisième jeu de données : 'eval' qui peut être utilisé pour l'évaluation finale du modèle.

Ces fichiers protocoles contiennent trois données : 
- le speaker id 'spk' c'est l'identifiant du locuteur, 
- l'utterance id 'utt' c'est l'identifiant unique de l'extrait audio
- le label informe si l'audio est réel ou synthétique.

---

## STEP 1
Notre Step1 s'occupe de : 

### Charger les données :
En créant les deux data frame, train_df et dev_df, contenant l'utterance id 'utt', le label 'is_spoof' et le chemin du fichier sur notre machine 'path'.
A l'aide des fonctions load_asvspoof_protocol et attach_paths_by_utt contenu dans le fichier src/data_utils. Ces fonctions permettent de charger les fichiers audio .flac et d'associer à chaque utt le chemin du fichier correspondant.
### Analyser les données :
En donnant les carastéristiques complètes de notre jeu de données :
Nombre total d'échantillons : 25380
Fichiers manquants : 0
Et la répartition entre les fichiers synthétiques (spoof) et réels (bonafide):
is_spoof
0     2580
1    22800
On observe un fort déséquilibre dans les données avec seulement ~10% d'audios bonafide.
### Extraire les coefs MFCC :
Pour préparer nos enregistrements audio au machine learning, on utilise la bibliothèque python Librosa, spécialisée dans le traitement de signaux audios. Elle permet de charger les fichiers .flac et de calculer les coefficients MFCC (Mel-Frequency Cepstral Coefficients). 
Les coefficients MFCC sont une des représentation de fichiers audios les plus utilisés en reconnaissance vocale.
Voilà comment l'encodage en coefficients MFCC fonctionne :
1) Après être chargé, chaque fichier audio est découpé en X fenêtres temporelles de 20 à 25ms chacune.
2) Pour chacune de ses fenêtres, Librosa calcul 13 MFCC qui encode 13 composantes de la transformée cepstrale de l'extrait audio qui représentent le timbre de la voix.
3) Pour pouvoir appliquer nos algorithmes on a besoin de vecteurs de dimension fixe. On calcul donc pour chaque audio la moyenne et l'écart type des fenêtres par MFCC. On se retrouve donc avec 13 moyennes et 13 écart types, soit un seul vecteur de 26 dimensions sur lequel on peut appliquer les algos.
Toutes ces étapes se font grâce à la fonction def extract_features.
On sauvegarde les vecteurs MFCC dans un dossier data pour la suite du projet.
### Entrainer les données avec l'algorithme SVM :
On a fais une baseline rapide de notre modèle sur 500 échantillons récupérés au hasard car il aurait été trop long de le faire sur tout le dataset.
Le modèle a montré les résultats suivant :
Accuracy : 0.93
AUC : 0.9256114130434783
Confusion matrix :
 [[  4  12]
 [  2 182]]
On voit que notre modèle a un biais pro-spoof à cause du déséquilibre important de notre dataset.

Remarque : étant toutes les deux absentes à cause de problèmes médicaux lors de la séance où les consignes ont été données, on a pris beaucoup de retard sur le rendu du step1 (tout ne marchait pas quand on a déposé le fichier sur dvl) et on n'a pas compris qu'il fallait faire une version par binômes. Désolées pour cette confusion et ce premier rendu incomplet.

---

## Step 2
Pour cette étape du projet on doit otpimiser et comparer plusieurs algorithmes de machine learning qu'on va appliquer sur les vecteurs MFCC sauvegardés dans le step 1.
Pour avoir une comparaison approfondie entre les différents modèles on a choisi de se répartir le travail comme suit :
Nora : modèles linéaires et à marge. On a pris en compte la nature des MFCC qui sont des caractéristiques basses dimensions donc adaptés aux modèles linéaires qui capturent la majeure partie de leur caractéristiques.
Maryam : modèles basés sur des arbres. On a choisi ces algorithmes car ils sont complémentaires aux modèles linéaires et permettent de capturer des caractéristiques plus complexes qu'on ne voit pas forcément avec les modèles linéaires ou quasi linéaires.

Pour chaque algorithme utilisé dans les différents step2, on a procédé comme suit :
1) Construction de la pipeline
2) Définition de la grille d'hyper-paramètres pour le Gridsearch
3) Mise en place du GridSearchCV
4) Entrainement du modèle

On a également rééquilibrer nos données à l'aide de 'class_weight="balanced"' qu'on a appliqué à tous nos algorithmes pour pallier le biais pro-spoof qu'on a observé lors du step1.

### Step 2 Nora : 
Les algorithmes qu'on a choisi dans cette version sont :
- SVM (Support Vector Machine) car il est résistant au bruit et efficace sur des vecteurs compacts.
- Régression Logistique car il est interprétable et permet d'avoir une baseline optimisée.
- Voting Classifier car il combine les atouts des deux algorithmes et permet d'améliorer la robustesse générale.

Pour les algorithmes SVM et régression logistique on a commencé par construire la pipeline du modèle en utilisant StandardScaler() pour normaliser les 26 caractéristiques MFCC.
Puis, on a défini la grille d'hyper paramètres avec :
Pour SVM : 
- la pénalisation de la marge C qui prend les valeurs 0.1, 1 et 10.
- le noyau (kernel) qui peut être linéaire (linear) ou non linéaire (rbf).
- l'impact de chaque point dans le noyau (gamma) qui prend les valeurs scale (l'impact dépend du nombres de caractéristiques et de leur variance) ou auto (il dépend seulement du nombre de caractéristiques).
Pour la régression logistique : 
- seulement la pénalisation de la marge C qui prend les mêmes valeurs que pour SVM.
Ensuite on a mis en place GridSearchCV puis on a entrainer les modèles pour qu'ils nous donnent le score AUC, le temps d'entrainement et les meilleurs hyperparamètres.
Puis on construit le voting classifier en utilisant le meilleur SVM obtenu via GridSearchCV et la meilleure régression logistique pour qu'il nous renvoi temps d'entrainement.

### Step 2 Maryam : 
Les algorithmes qu'on a choisi dans cette version sont :
- Decision Tree car ils permettent de capturer des interactions non linéaires entre les 26 caractéristiques des MFCC.
- Random Forest car il réduit fortement la variance et s'adapte mieux aux patterns complexes.
- Bagging Classifier car il renforce la stabilité du decision tree grâce à un ensemble.

Pour chaque algorithme on a construit la pipeline du modèle avec les paramètres à tester, puis on met en place le GridSearchCV, on entraine le modèle et on affiche les meilleurs hyperparamètres, le score AUC et le temps d'entrainement.
Pour le decision tree les paramètres qu'on test sont :
- criterion qui prend les valeurs gini (mesure la probabilité qu'un élément soit mal classé si on l'étiquette aléatoirement) ou entropy qui mesure le niveau de désordre du nœud.
- max_depth qui limite la profondeur pour éviter le sur-apprentissage, on l'a définit à None,5,10 et 20
- min_samples_split qui donne le nombre minimum d'exemples nécessaires pour diviser un noeud, définit à 2,5 et 10.
Pour le Random Forest : 
- n_estimators qui défini le nombre d’arbres dans la forêt qu'on a fixé à 100 et 300.
- max_depth qui donne profondeur maximale de chaque arbre, définit à None,10 et 20.
Pour BaggingClassifier on s'est basées sur le meilleur Decision Tree trouvé en lui donnant les paramètres :
- n_estimators = 20, on veut qu'il entraine 20 modèles en parallèle.
- max_samples = 0.8, chaque arbre est entrainé sur 80% des données tirées avec remise.
- bootstrap = True, active le tirage aléatoire avec remise.

### Fonction d'évaluation commune pour analyser les performances des modèles
On s'est accordées pour ajouter une fonction d'évaluation evaluate(name, model, X, y) qui calcule et renvoie le temps d'interférence, les probabilités prédites, les prédictions binaires et les principales métriques de classification.

---
>>>>>>> 7eae550 (Clean initial commit after adding .gitignore)
