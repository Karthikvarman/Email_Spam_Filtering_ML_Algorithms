This project uses Support Vector Machine (SVM), Logistic Regression, and Naïve Bayes classifiers to implement a machine learning-based email classification system that separates spam from ham. The following phases comprise the process that was 
followed:

1. Auto-labeling and Data Collection :-
   
Email messages make up the dataset, which is kept in spam_features_cleaned.csv. 
The system automatically labels the messages using a predefined list of popular spam 
keywords (such as win, free, cash, and urgent) if a label column is missing. These labels 
are kept as "spam" or "ham" in a new column. After that, the data is divided and stored as 
two distinct files, spam.csv and ham.csv.

3. Preprocessing Data :-
   
Labels are numerically encoded as follows to get the data ready for machine 
learning:  spam = 1, ham = 0. 
The dataset is divided into two sets: testing (20%) and training (80%). 
Using unigrams and bigrams, TF-IDF (Term Frequency-Inverse Document Frequency) 
vectorization is used to transform text messages into numerical feature vectors. This 
illustrates how crucial words and phrases are for spotting spam.

4. Training and Model Initialization :-

Three models for machine learning are set up: 
Support Vector Machine (SVM) with a linear kernel Logistic Regression with a 1000 
iteration limit. Multinomial Naïve Bayes is appropriate for discrete features such as word 
counts. The training set's TF-IDF vectors are used to train each model. By examining 
feature patterns in the data, these models are able to differentiate between spam and ham.

5. Model Assessment :-
   
The following metrics are used to assess each model on the test set: 
• Accuracy  
• Precision 
• Recall 
• F1 Score 
• Confusion Matrix 
• Classification Report

Heatmaps are used to visualize confusion matrices, and each quadrant—True Positives, 
False Positives, etc.—is described. Performance metrics for each model are plotted in a 
comparative bar chart. 
6. Selecting the Best Model :-
   
Based on highest accuracy and F1 score, the top-performing model is automatically 
chosen. After that, this model is applied to interactive real-time email message 
classification using user input. 
7. Interface for Real-time Spam Classification :-

Users can enter any custom email message into the system, and the best model 
chosen will process it. The input's TF-IDF vector is produced and categorized as Ham or 
Spam. 
Using three proven machine learning techniques, this methodology offers a thorough 
and comparative approach to spam detection. It places a strong emphasis on user 
interaction, evaluation, visualization, and preprocessing to guarantee efficacy and 
transparency in spam classification.
