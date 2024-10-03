import streamlit as st
import time
import re
import pickle
import hashlib
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import os
import json
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import requests
 
 
# Set page config
st.set_page_config(page_title="Quizify", page_icon="🧠", layout="wide")
 
# Custom CSS for styling
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@300;400;500;600&display=swap');
    .stApp {
        font-family: 'Playfair Display', serif;
    }
    .stApp, .stApp p {
        color: #004080 !important;
        font-family: 'Playfair Display', serif;
    }
    .stApp label {
        color: #f5e9ea !important;  
        font-weight: bold;        
        font-family: 'Playfair Display', serif;
    }
    .big-font {
        font-size: 3.5rem !important;
        color: #003366;
        text-align: center;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.5rem !important;
        color: #004080;
        text-align: center;
        margin-bottom: 2rem;
    }
    input {
        border-radius: 5px !important;
        background-color: #e0f7fa !important; /* Prominent color */
        color: #004080 !important;
        padding: 8px 16px !important; /* Reduced size */
        font-size: 1rem !important; /* Reduced font size */
        transition: all 0.3s ease;
    }
    input:focus {
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.5) !important; /* More prominent focus effect */
    }
   .stButton>button {
    color: #ffcc00 !important; /* Changed font color */
    border-radius: 10px;
    padding: 12px 30px;
    font-size: 20px; /* Increased font size */
    font-weight: 700; /* Made font weight bolder */
    transition: all 0.3s;
    border: none;
    text-transform: uppercase;
    letter-spacing: 1.5px; /* Slightly increased letter spacing */
    width: 100%;
    background-color: #88b1d8 ; /* Changed background color */
}

.stButton>button:hover {
    transform: translateY(-2px);
}

</style>


"""
 
# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)
 
def load_users():
    """Load user data from the pickle file."""
    try:
        with open('users.pkl', 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return {}
 
def save_users(users):
    """Save user data to the pickle file."""
    with open('users.pkl', 'wb') as f:
        pickle.dump(users, f)
 
def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()
 
def validate_password(password):
    """Check if password meets required criteria."""
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True
 
def main():
    st.markdown("<h1 class='big-font'>Welcome to Quizify</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Embark on a journey of knowledge with AI-generated quizzes!</p>", unsafe_allow_html=True)
 
    users = load_users()  # Load users from pickle file
 
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
 
    if st.session_state.authenticated:
        # If authenticated, show the dashboard
        st.sidebar.subheader("Dashboard")
        if st.sidebar.button("Dashboard"):
            st.page_link("pages/app.py" ,label="Generate your quiz")
    else:
        # Toggle form between login and signup
        if 'show_login' not in st.session_state:
            st.session_state.show_login = True
 
        if st.session_state.show_login:
            login(users)
        else:
            signup(users)
 
        # Toggle link
        toggle_text = "Don't have an account? Sign up" if st.session_state.show_login else "Already have an account? Log in"
        if st.button(toggle_text, key="toggle_link"):
            st.session_state.show_login = not st.session_state.show_login
 
 
 
def login(users):
    st.markdown("<h2 class='form-header'>Login to Your Account</h2>", unsafe_allow_html=True)
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_button"):
        if username in users and users[username] == hash_password(password):
            with st.spinner("Verifying your credentials..."):
                time.sleep(1.5)
            st.success("Welcome back! You're now logged in.")
            st.session_state.authenticated = True  
            st.rerun()  # Rerun the app to refresh the UI and state

        else:
            st.error("Invalid username or password.")
 
def signup(users):
    st.markdown("<h2 class='form-header'>Create Your Quizify Account</h2>", unsafe_allow_html=True)
    new_username = st.text_input("Choose a Username", key="signup_username")
    new_password = st.text_input("Create a Strong Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Your Password", type="password", key="signup_confirm")
 
    if st.button("Join Quizify", key="signup_button"):
        if new_username and new_password and confirm_password:
            if new_username in users:
                st.error("Username already exists. Please choose a different one.")
            elif new_password == confirm_password:
                if validate_password(new_password):
                    users[new_username] = hash_password(new_password)  # Save new user with hashed password
                    save_users(users)  # Save updated users to pickle file
                    with st.spinner("Creating your Quizify account..."):
                        time.sleep(1.5)
                    st.success("Welcome to Quizify! Your account has been created successfully.")
                else:
                    st.error("Password must be at least 8 characters long and contain uppercase, lowercase, digit, and special character.")
            else:
                st.error("Passwords do not match. Please try again.")
        else:
            st.error("Please fill in all fields to create your account.")
 
if __name__ == "__main__":
    main()