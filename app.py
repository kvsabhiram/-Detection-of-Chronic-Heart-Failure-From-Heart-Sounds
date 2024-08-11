from tkinter import *
from tkinter.filedialog import askdirectory, askopenfilename
import tkinter
import numpy as np
import os
import librosa
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

main = tkinter.Tk()
main.title("Machine Learning Models for Heart Disease Detection")
main.geometry("1300x1200")

global heart_X_train, heart_X_test, heart_y_train, heart_y_test
global heart_X
global heart_Y
global dataset_path
global model_rf
global model_dl
global model_recording

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def upload():
    global dataset_path
    dataset_path = askdirectory(initialdir=".")
    pathlabel.config(text=dataset_path)
    text.delete('1.0', END)
    text.insert(END, 'Heart disease audio dataset loaded\n')

def preprocess():
    text.delete('1.0', END)
    global heart_X
    global heart_Y
    global heart_X_train, heart_X_test, heart_y_train, heart_y_test

    heart_X = []
    heart_Y = []

    for subset in ['train', 'test']:
        for label in ['healthy', 'unhealthy']:
            folder = os.path.join(dataset_path, subset, label)
            label_value = 0 if label == 'healthy' else 1
            for file_name in os.listdir(folder):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(folder, file_name)
                    features = extract_features(file_path)
                    heart_X.append(features)
                    heart_Y.append(label_value)

    heart_X = np.array(heart_X)
    heart_Y = np.array(heart_Y)
    heart_X = normalize(heart_X)

    heart_X_train, heart_X_test, heart_y_train, heart_y_test = train_test_split(
        heart_X, heart_Y, test_size=0.2, random_state=0
    )

    text.insert(END, "Total records available in the dataset: " + str(heart_X.shape[0]) + "\n")
    text.insert(END, "Total records used to train machine learning algorithm (80%): " + str(heart_X_train.shape[0]) + "\n")
    text.insert(END, "Total records used to test machine learning algorithm (20%): " + str(heart_X_test.shape[0]) + "\n")

    # Plotting the distribution of the training dataset
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Training Data Distribution")
    plt.bar(['Healthy', 'Unhealthy'], [sum(heart_y_train == 0), sum(heart_y_train == 1)], color=['green', 'red'])
    plt.xlabel('Class')
    plt.ylabel('Count')

    # Plotting the distribution of the test dataset
    plt.subplot(1, 2, 2)
    plt.title("Test Data Distribution")
    plt.bar(['Healthy', 'Unhealthy'], [sum(heart_y_test == 0), sum(heart_y_test == 1)], color=['green', 'red'])
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()



def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return accuracy, sensitivity, specificity

def plot_metrics(accuracy, sensitivity, specificity, title):
    labels = ['Accuracy', 'Sensitivity (Recall)', 'Specificity']
    values = [accuracy, sensitivity, specificity]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['blue', 'orange', 'green'])
    plt.ylabel('Score')
    plt.title(f'{title} Metrics')
    plt.ylim(0, 1)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Unhealthy'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{title} Confusion Matrix')
    plt.show()

def run_ml_segmented_model():
    global model_rf
    text.delete('1.0', END)
    
    # Feature Extraction and Selection
    k_best = SelectKBest(f_classif, k=20)
    heart_X_train_selected = k_best.fit_transform(heart_X_train, heart_y_train)
    heart_X_test_selected = k_best.transform(heart_X_test)

    model_rf = RandomForestClassifier()
    model_rf.fit(heart_X_train_selected, heart_y_train)
    predict = model_rf.predict(heart_X_test_selected)
    accuracy, sensitivity, specificity = calculate_metrics(heart_y_test, predict)
    
    text.insert(END, f"Random Forest Classifier with Feature Selection:\n")
    text.insert(END, f"Accuracy: {accuracy}\n")
    text.insert(END, f"Sensitivity (Recall): {sensitivity}\n")
    text.insert(END, f"Specificity: {specificity}\n\n")
    
    # Plot metrics
    plot_metrics(accuracy, sensitivity, specificity, 'Random Forest')
    
    # Plot confusion matrix
    plot_confusion_matrix(heart_y_test, predict, 'Random Forest')

def run_dl_model():
    global model_dl
    text.delete('1.0', END)
    
    model_dl = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=500, alpha=0.0001,
                             solver='adam', verbose=10, random_state=21, tol=0.000000001)
    model_dl.fit(heart_X_train, heart_y_train)
    predict = model_dl.predict(heart_X_test)
    accuracy, sensitivity, specificity = calculate_metrics(heart_y_test, predict)
    
    text.insert(END, f"Deep Learning Model (MLP Classifier):\n")
    text.insert(END, f"Accuracy: {accuracy}\n")
    text.insert(END, f"Sensitivity (Recall): {sensitivity}\n")
    text.insert(END, f"Specificity: {specificity}\n\n")
    
    # Plot metrics
    plot_metrics(accuracy, sensitivity, specificity, 'Deep Learning')
    
    # Plot confusion matrix
    plot_confusion_matrix(heart_y_test, predict, 'Deep Learning')

def run_recording_ml_model():
    global model_recording
    text.delete('1.0', END)
    
    model_recording = XGBClassifier()
    model_recording.fit(heart_X_train, heart_y_train)
    predict = model_recording.predict(heart_X_test)
    accuracy, sensitivity, specificity = calculate_metrics(heart_y_test, predict)
    
    text.insert(END, f"Recording ML Model (XGB Classifier):\n")
    text.insert(END, f"Accuracy: {accuracy}\n")
    text.insert(END, f"Sensitivity (Recall): {sensitivity}\n")
    text.insert(END, f"Specificity: {specificity}\n\n")
    
    # Plot metrics
    plot_metrics(accuracy, sensitivity, specificity, 'XGB Classifier')
    
    # Plot confusion matrix
    plot_confusion_matrix(heart_y_test, predict, 'XGB Classifier')

# Example usage: Call one of the functions
# run_recording_ml_model()


def predict_chf():
    global model_rf, model_dl, model_recording
    
    if model_rf is None or model_dl is None or model_recording is None:
        text.insert(END, "Please train models first!\n")
        return
    
    text.delete('1.0', END)  # Clear previous text
    
    try:
        # Ask user to select a file for prediction
        filename = askopenfilename(initialdir=".")
        
        if filename:
            text.insert(END, f"Selected file: {filename}\n\n")
            
            # Extract features from the file
            features = extract_features(filename)
            features = features.reshape(1, -1)
            features = normalize(features)
            
            # Feature Selection for ML model (example using SelectKBest)
            k_best = SelectKBest(f_classif, k=20)
            features_selected = k_best.fit_transform(heart_X_train, heart_y_train)
            features_selected = k_best.transform(features)
            
            # Predictions using the models
            rf_predict = model_rf.predict(features_selected)
            dl_predict = model_dl.predict(features)
            recording_predict = model_recording.predict(features)
            
            # Voting mechanism
            predictions = [rf_predict[0], dl_predict[0], recording_predict[0]]
            final_prediction = max(set(predictions), key=predictions.count)
            
            # Display final prediction
            result = "Healthy" if final_prediction == 0 else "Unhealthy"
            text.insert(END, f"Final Prediction: {result}\n")
            
        else:
            text.insert(END, "No file selected.\n")
    
    except Exception as e:
        text.insert(END, f"Error predicting: {str(e)}\n")



def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Machine Learning Models for Heart Disease Detection')
title.config(bg='dark goldenrod', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Heart Disease Dataset", command=upload)
upload.place(x=700, y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=700, y=150)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=700, y=200)
preprocessButton.config(font=font1)

mlSegmentedButton = Button(main, text="Run ML Segmented Model with FE & FS", command=run_ml_segmented_model)
mlSegmentedButton.place(x=700, y=250)
mlSegmentedButton.config(font=font1)

dlModelButton = Button(main, text="Run DL Model on RAW Feature", command=run_dl_model)
dlModelButton.place(x=700, y=300)
dlModelButton.config(font=font1)

recordingModelButton = Button(main, text="Run Recording ML Model", command=run_recording_ml_model)
recordingModelButton.place(x=700, y=350)
recordingModelButton.config(font=font1)

predictButton = Button(main, text="Predict CHF from Test Sound", command=predict_chf)
predictButton.place(x=700, y=400)
predictButton.config(font=font1)

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=700, y=450)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=80)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=100)
text.config(font=font1)

main.config(bg='turquoise')
main.mainloop()
