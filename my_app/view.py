import streamlit as st

class View:
    def display_monitoring(self, reference_data, current_data):
        st.write("Displaying monitoring results...")

    def display_report(self, report, title):
        st.subheader(title)
        st.write(report)
