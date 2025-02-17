import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
import random

# File uploaders for datasets
train_file = st.file_uploader("Upload Training Data CSV", type=["csv"])
test_file = st.file_uploader("Upload Testing Data CSV", type=["csv"])

# Initialize session state variables
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None

if train_file and test_file:
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    if 'Unnamed: 133' in train_data.columns:
        train_data = train_data.drop(['Unnamed: 133'], axis=1)

    st.write("### Training Data Info")
    st.write(train_data.info())

    st.write("### Training Data Description")
    st.write(train_data.describe().T)

    st.write("### Missing Values")
    st.write(train_data.isna().sum())

    x_train = train_data.drop(["prognosis"], axis=1)
    y_train = train_data["prognosis"]
    x_test = test_data.drop(["prognosis"], axis=1)
    y_test = test_data["prognosis"]

    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(x_train.corr(), annot=False, linewidth=1, cmap='coolwarm')
    st.pyplot(fig)

    prognosis_counts = train_data['prognosis'].value_counts()
    fig, ax = plt.subplots(figsize=(10,12))
    ax.pie(prognosis_counts.values, labels=prognosis_counts.index, autopct='%1.1f%%')
    st.pyplot(fig)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    if st.button('Train SVM Model'):
        st.session_state.model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        st.session_state.model.fit(x_train_scaled, y_train)
        y_pred_svm = st.session_state.model.predict(x_test_scaled)
        ac_svm = accuracy_score(y_test, y_pred_svm)
        st.write(f"Accuracy score for SVM: {ac_svm}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_svm))

        cm = confusion_matrix(y_test, y_pred_svm)
        fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, colorbar=True, cmap='crest', figsize=(8, 8))
        st.pyplot(fig)
        
        # Set flag that model has been trained
        st.session_state.is_trained = True

    # Choose a random index from the dataset
    random_index = st.number_input("Enter Random Index for Testing", min_value=0, max_value=len(train_data)-1, value=0)  # Random index generation
    st.write(f"Randomly chosen index for prediction: {random_index}")

    if st.button('Test Random Prediction'):
        if st.session_state.is_trained:  # Check if model is trained
            random_row = train_data.loc[random_index]
            test_random = random_row.drop(["prognosis"])
            y_test_random = random_row["prognosis"]

            st.write(f"Predicted Disease: {st.session_state.model.predict([test_random])[0]}")
            st.write(f"Actual Disease: {y_test_random}")
        else:
            st.write("Please train the model first before making predictions.")