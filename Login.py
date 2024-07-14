import streamlit as st
import pandas as pd

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_icon="🧍‍♂️", page_title="LOGIN"
)

st.title("Welcome:- Parameter based health risk assessment and Disease outbreak forecasting")
st.title("LOGIN ")
import re

st.markdown(
    """
    <style>
    .fullScreenFrame {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 10vh;
    }
    section{
        background-color: #cccccc;
        background-image: linear-gradient(#D4CD82, #8e9ebf);
    }
    [data-testid="stHeader"]{
        background-color:#dbdaa0;
    }
    </style>
    
    """,

    unsafe_allow_html=True
)

# To hide side bar we use :-
st.markdown(
    """
        <style>
    [data-testid="stSidebar"]{
        visibility: hidden;
    }
    [data-testid="stSidebarNavLink"]{
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)
df = pd.read_csv("data.csv")
data = pd.DataFrame(df)

st.markdown(
    """
    <style>
    .input-label {
        font-size: 20px;  /* Adjust this value as needed */
        font-weight: bold;
        margin-bottom: -40px; /* Adjust or remove this if not needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<p class='input-label'>Enter email</p>", unsafe_allow_html=True)
email = st.text_input("", key="email")

st.markdown("<p class='input-label'>Enter Password</p>", unsafe_allow_html=True)
password = st.text_input("", type="password", key="password")

# Center-align the form and login button
st.markdown('<div class="centered-content">', unsafe_allow_html=True)

# Check if the user clicked the login button
try:
    if st.button("Login"):

        if data[data["Email"] == email]["Password"].values[0] == password:
            st.success("Login successful")
            st.markdown(f'<meta http-equiv="refresh" content="2;url=http://localhost:8501/main">',
                        unsafe_allow_html=True)
            st.header("Redirecting...")
        else:
            st.error("Invalid Email Or Password")
except:
    st.warning("Enter email And Password")

col1, col2 = st.columns([0.5, 0.5])

# Button for registration
if col1.button("Register"):
    st.markdown('<meta http-equiv="refresh" content="0;url=http://localhost:8501/Signup">',
                unsafe_allow_html=True)

# Close the centered-content div
st.markdown('</div>', unsafe_allow_html=True)
