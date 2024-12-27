import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

# Load the dataset
data = pd.read_csv('lung cancer data.csv')

# Handle missing values (if any)
data.dropna(inplace=True)

# Debug: Check class distribution
st.write("Class Distribution:")
st.write(data['LUNG_CANCER'].value_counts())

# Encode categorical data
label_encoders = {}
for column in ['GENDER', 'SMOKING', 'LUNG_CANCER']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the dataset
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True, class_weight='balanced')
}

best_model = None
best_f1 = 0
st.write("Model Training Metrics:")
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    st.write(f"{model_name}: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# Save the best model and scaler
joblib.dump(best_model, 'lung_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load the trained model and scaler
model = joblib.load('lung_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App
st.title("Lung Cancer Prediction App")

st.sidebar.header("User  Input Features")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", options=label_encoders['GENDER'].classes_)
smoking = st.sidebar.selectbox("Smoking", options=label_encoders['SMOKING'].classes_)

# Predict button
if st.sidebar.button("Predict"):
    try:
        # Encode user inputs
        gender_encoded = label_encoders['GENDER'].transform([gender])[0]
        smoking_encoded = label_encoders['SMOKING'].transform([smoking])[0]

        # Prepare full feature set
        input_features = X.iloc[0].copy()  # Copy an example row for proper alignment
        input_features["AGE"] = age
        input_features["GENDER"] = gender_encoded
        input_features["SMOKING"] = smoking_encoded

        # Preprocess the input data
        features_scaled = scaler.transform([input_features])
        
        # Make prediction
        prediction = model.predict(features_scaled)
        output = 'Yes' if prediction[0] == 1 else 'No'
        st.success(f"Prediction: Lung Cancer - {output}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display dataset statistics
if st.checkbox("Show Dataset Summary"):
    st.write(data.describe())

# Show visualizations
if st.checkbox("Show Target Variable Distribution"):
    st.subheader("Distribution of Lung Cancer Cases")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='LUNG_CANCER', data=data)
    st.pyplot(plt.gcf())
    plt.clf()

if st.checkbox("Show Correlation Heatmap"):
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    st.pyplot(plt.gcf())
    plt.clf()
