import streamlit as st

class View:
    @staticmethod
    def display_report(report, report_name):
        """Display a report in the Streamlit app."""
        st.subheader(report_name)
        
        if isinstance(report, dict):
            st.write(report)
        else:
            # If the report is an Evidently Report object, use the show method
            report.show()
            
    @staticmethod
    def display_monitoring(reference_data, current_data):
        """Display monitoring data."""
        st.subheader("Reference Data")
        st.write(reference_data)

        st.subheader("Current Data")
        st.write(current_data)
