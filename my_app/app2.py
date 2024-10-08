import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from model import Model
from view import View

# Load the Iris dataset directly from sklearn
@st.cache_resource
def load_iris_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y  # Add target to DataFrame for monitoring
    return df, y, target_names

def main():
    st.title("Iris Classification and Monitoring App")

    # Load dataset
    df, y, target_names = load_iris_data()

    # Initialize model and view
    model = Model()
    view = View()

    # Add species name to the dataframe for EDA purposes
    df['species'] = pd.Series(y).map({i: target_names[i] for i in range(len(target_names))})
    # Perform Exploratory Data Analysis (EDA)
    view.display_monitoring(df, df)  # Display the same data for monitoring purposes (you may change this)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target', 'species']), y, test_size=0.2, random_state=42)
    
    # Prepare data for monitoring
    reference_data = df.drop(columns=['species']).copy()  # Use full dataset for reference
    current_data = df.drop(columns=['species']).copy()    # For demo purposes, this can be changed

    # Train model
    model.train_model(reference_data, current_data)

    # Monitoring Section
    st.subheader("Monitoring Options")
    if st.button("Generate Reports"):
        performance_report = model.performance_report(reference_data, current_data)
        view.display_performance_report(performance_report, "Model Performance Report")

        target_report = model.target_report(reference_data, current_data)
        view.display_report(target_report, "Target Drift Report")

        data_drift_report = model.data_drift_report(reference_data, current_data)
        view.display_report(data_drift_report, "Data Drift Report")

        data_quality_report = model.data_quality_report(reference_data, current_data)
        view.display_report(data_quality_report, "Data Quality Report")

if __name__ == "__main__":
    main()
