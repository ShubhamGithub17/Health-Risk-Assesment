import streamlit as st
import pandas as pd
import re
import os.path

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_icon="🧑‍⚕️", page_title=""
)

# Hide sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{
        visibility: hidden;
    }
    section{
        background-color: #e6b48e;
    }
    [data-testid="stHeader"]{
        background-color: #e6b48e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create Streamlit form for registration
st.title('Registration Form')

# Check if CSV file exists, if not, create it with headers
if not os.path.isfile("data.csv"):
    data = {"Name": [], "Email": [], "Mobile": [], "Password": [], "Confirm_Password": []}
    pd.DataFrame(data).to_csv("data.csv", index=False)

# Collect data from the user
name = st.text_input("**Enter Full Name**")
email = st.text_input("**Enter Email**")
mobile = st.text_input("**Enter mobile number**")
password = st.text_input("**Enter Password**", type="password")
cpassword = st.text_input("**Confirm Password**", type="password")


# When the user submits the form
def handle_submit():
    # Check if passwords match
    if password != cpassword:
        st.error('Passwords do not match. Please re-enter.')
    # Verify email format
    elif not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
        st.error('Please enter a valid email address.')
    elif not re.match(r'^[0-9]{10}$', mobile):
        st.error('Please enter a valid 10-digit mobile number.')
    # Verify password strength (e.g., minimum length)
    elif len(password) < 8:
        st.error('Password should be at least 8 characters long.')
    else:
        # Load existing data
        existing_data = pd.read_csv("data.csv")
        # Check if email is already registered
        if email in existing_data['Email'].values:
            st.error('Email address is already registered. Please use a different email.')
        else:
            # Save data to CSV
            data = {"Name": [name], "Email": [email], "Mobile": [mobile], "Password": [password],
                    "Confirm_Password": [cpassword]}
            df = pd.DataFrame(data)
            df.to_csv("data.csv", index=False, mode='a', header=False)
            # Display success message
            st.success("Sign up successful!")
            st.markdown(f'<meta http-equiv="refresh" content="2;url=http://localhost:8501/Login">',
                        unsafe_allow_html=True)
            st.header("Redirecting...")


col1, col2 = st.columns([0.1, 0.5])

if col1.button("Submit"):
    handle_submit()

# Login button
if col2.button("Login"):
    st.markdown('<meta http-equiv="refresh" content="0;url=http://localhost:8501/Login">',
                unsafe_allow_html=True)