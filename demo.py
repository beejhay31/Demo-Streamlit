import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import sweetviz as sv


# Placeholder for user authentication (SSO simulation)
def authenticate_user():
    # In a real app, implement SSO logic here (e.g., with OAuth)
    return True  # Simulate successful authentication

# Load the Iris dataset
@st.cache_resource
def load_iris_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return pd.DataFrame(X, columns=feature_names), pd.Series(y), target_names

# Train a logistic regression model
@st.cache_resource
def train_model(X, y):
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model

# Detect model drift
def detect_drift(reference_data, current_data):
    # Define the column mapping
    column_mapping = ColumnMapping()

    # Create a report object
    report = Report(metrics=[
        DataDriftPreset(),
    ])

    # Generate the report
    report.run(reference_data, current_data, column_mapping)
    
    # Save the report as HTML
    report.save('drift_report.html')

    # Read and display the report
    with open('drift_report.html', 'r') as f:
        drift_report_html = f.read()
    
    st.components.v1.html(drift_report_html, height=600)

# Streamlit app structure
def main():
    st.title("Iris Classification App")

    # Authenticate user
    if authenticate_user():
        st.success("User authenticated successfully!")

        # Load dataset
        df, y, target_names = load_iris_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

        # Train model
        model = train_model(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, predictions, target_names=target_names))

        # Run EDA using AutoViz
        st.subheader("Exploratory Data Analysis (EDA) with AutoViz")
        analyze_report = sv.analyze(df, target_feat='target')
        analyze_report.show_html(report.html', open_browser=False)
        #auto_viz = AutoViz_Class()
        #auto = auto_viz.AutoViz(df, depVar='target')
        #auto.show()

        # Simulate loading current dataset for drift detection
        # In a real scenario, this would come from a new data source
        current_data = df.copy()  # Here, we just reuse the original data for demonstration
        detect_drift(df, current_data)

if __name__ == "__main__":
    main()
