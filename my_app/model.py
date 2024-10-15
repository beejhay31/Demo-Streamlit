import pandas as pd
import joblib
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metric_preset.regression_performance import RegressionPreset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

class Model:
    def __init__(self):
        self.model = LogisticRegression(max_iter=200)
        self.column_mapping = ColumnMapping()
        self.column_mapping.target = 'target'
        self.column_mapping.prediction = 'prediction'
        self.column_mapping.numerical_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    def train_model(self, reference_data, current_data):
        st.write("Training the logistic regression model...")
        X_train = reference_data.drop(columns=['target'])
        y_train = reference_data['target']
        
        self.model.fit(X_train, y_train)

        # Model training complete
        st.write("Model training complete.")

    """def predict(self, X):
        # Expose the predict method of the LogisticRegression model
        return self.model.predict(X)"""

    def performance_report(self, reference_data, current_data):
        st.write("Generating model performance report...")
        
        X_test = current_data.drop(columns=['target'])
        y_test = current_data['target']
        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        #report = classification_report(y_test, predictions, target_names=target_names)
        """performance_report = {
            'accuracy': accuracy,
            'classification_report': report
        }""" 

        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.text(report)
        #return performance_report


    def target_report(self, reference_data, current_data):
        st.write("Generating Target Drift Report...")
        target_drift_report = Report(metrics=[TargetDriftPreset()])
        target_drift_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        return target_drift_report

    def data_drift_report(self, reference_data, current_data):
        st.write("Generating Data Drift Report...")
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        return data_drift_report

    def data_quality_report(self, reference_data, current_data):
        st.write("Generating the Data Quality Report will take more time, around 10 minutes, due to its thorough analysis.")
        data_quality_report = Report(metrics=[DataQualityPreset()])
        data_quality_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        return data_quality_report
