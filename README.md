## Audio deepfake detection project

Our project aims to implement a complete pipeline for detecting audio deepfakes using machine learning algorithms.
To achieve this, we train our model on the ASVspoof 2019 dataset. This dataset contains two types of audio files: PAs (Physical Addresses) and LAs (Logical Addresses). We will use LAs for our project because they are AI-generated audio files. PAs are replays of existing recordings, and therefore not useful for our project. For the two first steps, we will use two subsets provided with the dataset: `train` for model training and `dev` for validation and evaluation.

---

## Presentation of the ASVspoof 2019-LA dataset

The ASVspoof 2019-LA dataset contains several labeled .flac audio files, which can be either AI-generated synthetic audio (spoof) or actual audio recordings (bonafide).
These audio files are divided into two categories: training data ('train') and evaluation data ('dev'). They are labeled in the protocol files ASVspoof2019.LA.cm.train.trn.txt (training) and ASVspoof2019.LA.cm.dev.trl.txt.
The dataset also provides a third dataset, 'eval', which can be used for the final evaluation of the model.

These protocol files contain three pieces of data:
- the speaker ID 'spk', which is the speaker's identifier,
- the utterance ID 'utt', which is the unique identifier of the audio excerpt,
- the label, which indicates whether the audio is real or synthetic.

---

## STEP 1
Our Step1 is in charge of : 

### Loading the data :
By creating two data frames, `train_df` and `dev_df`, containing the utterance ID 'utt', the label 'is_spoof', and the file path on our machine 'path'.
Using the `load_asvspoof_protocol` and `attach_paths_by_utt` functions contained in the `src/data_utils` file. These functions allow us to load the .flac audio files and associate each utterance ID with the corresponding file path.
### Analysing the data :
Providing the complete characteristics of our dataset:
Total number of samples: 25380
Missing files: 0
And the breakdown between synthetic (spoof) and real (bonafide) files:
is_spoof
0     2580
1    22800
We observe that there is a strong imbalance in the data with only ~10% of bonafide audios.
### Extracting the MFCC coefficients :
To prepare our audio recordings for machine learning, we use the Librosa Python library, which specializes in audio signal processing. It allows us to load .flac files and calculate MFCC (Mel-Frequency Cepstral Coefficients).
MFCCs are one of the most widely used representations of audio files in speech recognition.
Here's how MFCC encoding works:
1) After being loaded, each audio file is divided into X time windows of 20 to 25 ms each.
2) For each of these windows, Librosa calculates 13 MFCCs that encode 13 components of the cepstral transform of the audio clip, representing the timbre of the voice.
3) To apply our algorithms, we need vectors of fixed dimensions. Therefore, for each audio file, we calculate the mean and standard deviation of the windows using MFCCs. We are thus left with 13 means and 13 standard deviations, forming a single 26-dimensional vector on which we can apply the algorithms.
All these steps are performed using the `def extract_features` function.
We save the MFCC vectors in a `data` folder for later use in the project.
### Training the data using the SVM algorithm :
We performed a quick baseline test of our model on 500 randomly selected samples because it would have been too time-consuming to run it on the entire dataset.
The model showed the following results:
Accuracy : 0.93
AUC : 0.9256114130434783
Confusion matrix :
 [[  4  12]
 [  2 182]]
We can see that our model has a pro-spoof bias due to the significant imbalance in our dataset.

Note: As we were both absent due to medical issues during the session where the instructions were given, we fell far behind on submitting step 1 (not everything was working when we uploaded the file to dvl) and we didn't realize that we needed to create a version in pairs. We apologize for this confusion and the incomplete first submission.

---

## Step 2
For this step of the project, we need to optimize and compare several machine learning algorithms that we will apply to the MFCC vectors saved in step 1.
To obtain a thorough comparison between the different models, we chose to divide the work as follows:
Nora: linear and margin models. We took into account the nature of MFCCs, which are low-dimensional features and therefore well-suited to linear models that capture most of their characteristics.
Maryam: tree-based models. We chose these algorithms because they are complementary to linear models and allow us to capture more complex features that are not necessarily visible with linear or quasi-linear models.

For each algorithm used in the different step 2s, the following procedure was followed:
1) Pipeline construction
2) Definition of the hyperparameter grid for Gridsearch
3) Implementation of GridSearchCV
4) Model training

We also rebalanced our data using 'class_weight="balanced"' which we applied to all our algorithms to address the pro-spoof bias we observed during step 1.

### Step 2 Nora : 
The algorithms chosen for this version are:
- Support Vector Machine (SVM) because it is noise-resistant and efficient on compact vectors.
- Logistic Regression because it is interpretable and allows for an optimized baseline.
- Voting Classifier because it combines the strengths of both algorithms and improves overall robustness.

For the SVM and logistic regression algorithms, we began by building the model pipeline using StandardScaler() to normalize the 26 MFCC features.
Then, we defined the hyperparameter grid with:

For SVM:
- the margin penalty C, which takes the values 0.1, 1, and 10.
- the kernel, which can be linear or nonlinear (rbf).
- the impact of each point in the kernel (gamma), which takes the values scale (the impact depends on the number of features and their variance) or auto (it depends only on the number of features).

For logistic regression:
- only the penalty of margin C, which takes the same values as for SVM.

Next, we implemented GridSearchCV and trained the models to obtain the AUC score, training time, and best hyperparameters.
Then, we built the voting classifier using the best SVM obtained via GridSearchCV and the best logistic regression to return the training time.

### Step 2 Maryam : 
The algorithms chosen for this version are:
- Decision Tree because they allow us to capture non-linear interactions between the 26 features of MFCCs.
- Random Forest because it significantly reduces variance and adapts better to complex patterns.
- Bagging Classifier because it enhances the stability of the decision tree using an ensemble.
For each algorithm, we built the model pipeline with the parameters to be tested. Then, we set up the GridSearchCV, trained the model, and displayed the best hyperparameters, the AUC score, and the training time.

For the decision tree, the parameters we test are:
- criterion, which takes the values ​​of gini (measuring the probability that an element will be misclassified if randomly labeled) or entropy, which measures the level of disorder of the node.
- max_depth, which limits the depth to avoid overfitting; we defined it as None, 5, 10, and 20.
- min_samples_split, which gives the minimum number of examples needed to split a node; we defined it as 2, 5, and 10.

For the Random Forest:
- n_estimators defines the number of trees in the forest, set to 100 and 300.
- max_depth gives the maximum depth of each tree, set to None, 10, and 20.

For BaggingClassifier, we based our approach on the best Decision Tree found, giving it the following parameters:
- n_estimators = 20, we want it to train 20 models in parallel.
- max_samples = 0.8, each tree is trained on 80% of the data drawn with replacement.
- bootstrap = True, enables random sampling with replacement.

### Common evaluation function for analyzing model performance
We agreed to add an evaluation function evaluate(name, model, X, y) which calculates and returns the interference time, predicted probabilities, binary predictions and the main classification metrics.
We load the two selected models (best **SVM** and best **Random Forest** saved as `.pkl`), extract the same **26-dim MFCC features** for the eval audios, and compute standard metrics (**Accuracy, F1-score, ROC-AUC**) along with the **confusion matrix**. This final evaluation checks how well the trained models generalize to a harder split with spoofing conditions not used during development.

### Note : both step 2 are in the 'notebook' folder
---
>>>>>>> 7eae550 (Clean initial commit after adding .gitignore)
