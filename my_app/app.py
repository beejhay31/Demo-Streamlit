import streamlit as st
from classification import load_iris_data, perform_eda, train_model, model_performance
from monitoring import Monitoring
from model import Model
from sklearn.model_selection import train_test_split
import random
import smtplib  # For sending email

# Function to send the login code via email
def send_login_code(email, code):
    print(f"Login code for {email}: {code}")  # Replace with actual email sending logic using smtplib or an API like SendGrid

# Initialize session state for login management
if 'login_code' not in st.session_state:
    st.session_state['login_code'] = None
if 'email' not in st.session_state:
    st.session_state['email'] = None
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Login flow
def login():
    st.title("Login")

    email = st.text_input("Enter your email")
    if email and email.endswith("@datadock.ai"):
        if st.button("Send Login Code"):
            code = random.randint(100000, 999999)
            st.session_state['login_code'] = code
            st.session_state['email'] = email
            send_login_code(email, code)
            st.success("A login code has been sent to your email.")

    elif email and not email.endswith("@datadock.ai"):
        st.error("Please use a @datadock.ai email.")

    if st.session_state['login_code'] is not None:
        entered_code = st.text_input("Enter the code sent to your email", type='password')
        if st.button("Verify Code"):
            if int(entered_code) == st.session_state['login_code']:
                st.session_state['logged_in'] = True
                st.success("You are now logged in.")
            else:
                st.error("Invalid code. Please try again.")

# Main app interface after login
def main_app():
    st.title("Iris Classification and Monitoring App")

    view = View()
    model = Model()
    monitoring = Monitoring(view, model)

    # Add option to select between classification or monitoring
    option = st.sidebar.selectbox('Select Application', ('Exploratory Data Analysis (EDA)', 'Classification', 'Monitoring'))

    if option == 'Exploratory Data Analysis (EDA)':
        st.write("EDA Interface")
        df, y, target_names = load_iris_data()
        df['species'] = y.map({i: target_names[i] for i in range(len(target_names))})

        perform_eda(df)

    elif option == 'Classification':
        st.write("Classification Interface")
        df, y, target_names = load_iris_data()
        df['species'] = y.map({i: target_names[i] for i in range(len(target_names))})

        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['species']), y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)

        model_performance(model, X_test, y_test, target_names)

    elif option == 'Monitoring':
        st.write("Monitoring Interface")
        monitoring.run_monitoring()

# Main Streamlit app logic
if __name__ == '__main__':
    if not st.session_state['logged_in']:
        login()
    else:
        main_app()
