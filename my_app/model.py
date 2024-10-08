import streamlit as st

class Model:
    def train_model(self, reference_data, current_data):
        st.write("Training model on selected data...")

    def performance_report(self, reference_data, current_data):
        return "Model Performance Report"

    def target_report(self, reference_data, current_data):
        return "Target Drift Report"

    def data_drift_report(self, reference_data, current_data):
        return "Data Drift Report"

    def data_quality_report(self, reference_data, current_data):
        return "Data Quality Report"
