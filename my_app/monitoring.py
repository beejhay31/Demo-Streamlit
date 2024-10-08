import streamlit as st
import pandas as pd
import time

# Monitoring functions
class Monitoring:
    def __init__(self, view, model):
        self.view = view
        self.model = model

    def run_monitoring(self):
        st.title("Data & Model Monitoring App")
        st.write("Select the Date and month range from the sidebar and click 'Submit' to start monitoring.")

        # Date and month range selection
        new_start_month = st.sidebar.selectbox("Start Month", range(1, 7), 1)
        new_end_month = st.sidebar.selectbox("End Month", range(1, 7), 1)
        new_start_day = st.sidebar.selectbox("Start Day", range(1, 32), 1)
        new_end_day = st.sidebar.selectbox("End Day", range(1, 32), 30)
        
        # Report selection
        st.subheader("Select Reports to Generate")
        generate_model_report = st.checkbox("Generate Model Performance Report")
        generate_target_drift = st.checkbox("Generate Target Drift Report")
        generate_data_drift = st.checkbox("Generate Data Drift Report")
        generate_data_quality = st.checkbox("Generate Data Quality Report")

        if st.button("Submit"):
            st.write("Fetching current batch data...")
            data_start = time.time()
            df = pd.read_csv("data/Monitoring_data.csv")
            data_end = time.time()
            time_taken = data_end - data_start
            st.write(f"Fetched the data in {time_taken:.2f} seconds")

            date_range = (
                (df['Month'] >= new_start_month) & (df['DayofMonth'] >= new_start_day) &
                (df['Month'] <= new_end_month) & (df['DayofMonth'] <= new_end_day)
            )
            reference_data = df[~date_range]
            current_data = df[date_range]

            # Display and train model
            self.view.display_monitoring(reference_data, current_data)
            self.model.train_model(reference_data, current_data)

            # Generate reports based on selected checkboxes
            if generate_model_report:
                performance_report = self.model.performance_report(reference_data, current_data)
                self.view.display_report(performance_report, "Model Performance Report")

            if generate_target_drift:
                target_report = self.model.target_report(reference_data, current_data)
                self.view.display_report(target_report, "Target Drift Report")

            if generate_data_drift:
                data_drift_report = self.model.data_drift_report(reference_data, current_data)
                self.view.display_report(data_drift_report, "Data Drift Report")

            if generate_data_quality:
                data_quality_report = self.model.data_quality_report(reference_data, current_data)
                self.view.display_report(data_quality_report, "Data Quality Report")
