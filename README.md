
# Lung Cancer Prediction App

This project is a machine learning-based web application for predicting lung cancer using various features such as age, gender, and smoking history. The application uses a trained model to classify whether a person has lung cancer based on input features.

## Project Overview

This web application allows users to input features like age, gender, and smoking history to predict the likelihood of having lung cancer. The model has been trained using a dataset containing demographic and health-related information. The model uses machine learning classifiers, such as Logistic Regression, Decision Tree, Random Forest, and SVM, and selects the best model based on performance metrics.

## Features
- **User Input**: The app provides a sidebar for users to input their age, gender, and smoking history.
- **Prediction**: The model predicts whether the user is likely to have lung cancer based on the input features.
- **Model Evaluation**: The app trains multiple machine learning models and displays their performance metrics such as accuracy, precision, recall, and F1-score.
- **Visualization**: It provides options to visualize the target variable distribution and a correlation heatmap of the features in the dataset.
- **Model Persistence**: The best model is saved using joblib, which allows for loading and making predictions.

## Technologies Used
- **Python**: The main programming language used for data processing and model training.
- **Streamlit**: A Python library for building interactive web apps.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For training and evaluating machine learning models.
- **Matplotlib & Seaborn**: For creating visualizations.
- **Joblib**: For saving and loading the trained model.

## Dataset

The dataset used for training the model contains various features that can influence the likelihood of developing lung cancer. Features include:

- **AGE**: The age of the individual.
- **GENDER**: The gender of the individual.
- **SMOKING**: Whether the individual has a smoking history (Yes/No).
- **LUNG_CANCER**: The target variable, which indicates if the individual has lung cancer (1 for yes, 0 for no).

The dataset can be found as `lung cancer data.csv`.

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/lung-cancer-prediction.git
    cd lung-cancer-prediction
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:
      ```bash
      venv\Scripts\activate
      ```

    - On MacOS/Linux:
      ```bash
      source venv/bin/activate
      ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    You can create a `requirements.txt` file with the following content:

    ```
    pandas
    scikit-learn
    streamlit
    matplotlib
    seaborn
    joblib
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Open the application in your browser by navigating to the URL provided in the terminal, typically `http://localhost:8501`.

3. Enter the required features in the sidebar and click "Predict" to get the lung cancer prediction.

## How it Works

- **Model Training**: The dataset is preprocessed by handling missing values and encoding categorical variables. Then, it is split into training and test sets. Multiple machine learning models (Logistic Regression, Decision Tree, Random Forest, and SVM) are trained on the training data. The model with the highest F1-score is selected as the best model.
  
- **Prediction**: After the model is selected, the trained model is saved as a `.pkl` file. The Streamlit app allows users to input their age, gender, and smoking history, which are then encoded and scaled. The model predicts whether the user is likely to have lung cancer based on these inputs.

## Model Evaluation

The following metrics are used to evaluate the models:
- **Accuracy**: The proportion of correctly classified instances.
- **Precision**: The proportion of true positives out of all positive predictions.
- **Recall**: The proportion of true positives out of all actual positives.
- **F1 Score**: The harmonic mean of precision and recall.

The model with the highest F1-score is saved as the best model.
