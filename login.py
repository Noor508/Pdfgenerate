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
 
    [data-testid="stSidebar"] { display: none;}
    </style>
 
 
    """
# [data-testid="stSidebar"] { display: none;}
 
st.markdown(custom_css, unsafe_allow_html=True)
 
def save_user_results(username, quiz_title, score):
    try:
        user_results = load_user_results()
        user_results[username] = user_results.get(username, []) + [(quiz_title, score)]
        with open('user_data.pkl', 'wb') as f:
            pickle.dump(user_results, f)
    except Exception as e:
        print(f"Error saving user results: {e}")
 
# Function to load user quiz results from the pickle file
def load_user_results():
    try:
        with open('user_data.pkl', 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return {}
 
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
        st.switch_page("pages/app.py")
    else:
        # Toggle form between login and signup
        if 'show_login' not in st.session_state:
            st.session_state.show_login = True
 
        if st.session_state.show_login:
            login(users)
 
 
def login(users):
    row_input = st.columns((2, 1, 2, 1))
    with row_input[0]:
        username = st.text_input('Username')
 
        # Password visibility toggle logic
        if 'password_visible' not in st.session_state:
            st.session_state.password_visible = False
 
        password_type = "text" if st.session_state.password_visible else "password"
        password = st.text_input('Password', type=password_type)
 
       
 
        if st.button("Login"):
            if username in users and users[username] == hash_password(password):
                with st.spinner("Verifying your credentials..."):
                    time.sleep(1.5)
                st.success("Welcome back! You're now logged in.")
                st.session_state.authenticated = True  
                st.session_state.username = username  
                st.rerun()  
            else:
                st.error("Invalid username or password.")

 
if __name__ == "__main__":
    main()
 