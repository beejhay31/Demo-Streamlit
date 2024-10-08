import streamlit as st
from classification import load_iris_data, perform_eda, train_model, model_performance
from monitoring import Monitoring
from model import Model
from view import View
from sklearn.model_selection import train_test_split

# Predefined valid 7-digit login codes
VALID_CODES = ["1234567", "2345678", "3456789", "4567890", "5678901", "6789012", "7890123"]

# Initialize session state for login management
if 'login_code' not in st.session_state:
    st.session_state['login_code'] = None
if 'email' not in st.session_state:
    st.session_state['email'] = None
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Login flow using predefined unique 7-digit codes and valid email
def login():
    st.title("Login")

    # Email input: only allow emails with @datadock.ai domain
    email = st.text_input("Enter your email")
    
    # Validate email domain
    if email:
        if not email.endswith("@datadock.ai"):
            st.error("Invalid email domain. Only @datadock.ai email addresses are allowed.")
        else:
            st.session_state['email'] = email
            st.success(f"Welcome, {email}. Please enter your 7-digit login code.")
    
    # Code input and verification
    entered_code = st.text_input("Enter the 7-digit login code", type='password')

    if st.button("Verify Code"):
        if entered_code in VALID_CODES:
            st.session_state['login_code'] = entered_code
            st.session_state['logged_in'] = True
            st.success("You are now logged in. Redirecting to the app...")
        else:
            st.error("Invalid login code. Please try again.")

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
        login()  # If not logged in, show the login screen
    else:
        main_app()  # If logged in, show the main app interface
