test_text = """
welcome back to another machine learning explained video by assembly ai in this video we talk about supervised learning which is arguably the most important type of machine learning you will learn what it means examples of supervised learning or this data and training types of supervised learning and we touch on specific algorithms of supervised learning let's begin with the very basics what does machine learning mean machine learning is a sub-area of artificial intelligence and it's the study of algorithms that give computers the ability to learn and make decisions based on data and not from explicit instructions a popular example is learning to predict whether an email is spam or no spam by reading many different emails of these two types we typically differentiate between three types of machine learning supervised learning unsupervised learning and reinforcement learning in supervised learning the computer learns by making use of labeled data so we know the corresponding label or target of our data an example is again the spam prediction algorithm where we show many different emails to the computer and for each email we know if this was a spam email or not on the other hand in unsupervised learning the computer learns by making use of unlabeled data so we have data but we don't know the corresponding target an example is to cluster books into different categories on the basis of the title and other book information but not by knowing its actual category and then there is also reinforcement learning where so-called intelligent software agents take actions in an environment and automatically try to improve its behavior this usually works with a system of rewards and punishments and popular examples are games for example a computer can learn to be good in the snake game only by playing the game itself and every time
it eats an apple or it dies it learns from this actions now in this video we are going to focus on supervised learning where we learn from labeled data now what is data data can be any relevant information we collect for our algorithm this can be for example user information like age and gender or text data or images or information within an image like measurements or color information the possibilities are endless here let's look at a concrete example in the popular iris flower data set we want to predict the type of iris flower based on different measurements we have 150 records of flowers with different attributes that have been measured before so for each flower we have the sepal
length saypal width petal length and petal width these are called the features and we also have the corresponding species this is called the class the label or the target so this is a supervised case where we know the label we can
represent this table in a mathematical way so we put each feature into a vector this is the feature vector and then we do this for all the different samples and when we do this for all the different samples we end up in a 2d representation which is also called a matrix additionally we can put all labels into one vector this is called the target vector now in supervised learning we take the features and the labels and show it to the computer so that it learns we call this the training step and the data we use is called the training data training is performed by specific algorithms that usually try to minimize an error during this training process and this is done by mathematical optimization methods which i won't go into more detail here after training we want to show new data to the computer that it has never seen before and where we don't know the label this is called our test data and now the trained computer should be able to make a decision based on the information it has seen and determine the correct target value and this is how supervised learning works there are two types of supervised learning classification and regression in classification we predict a discrete class label in the previous flower classification example our target values can only have the values 0 1 and 2 corresponding to the three different classes if we have more than two possible labels like here we call this a multi-class classification problem if we only have two labels usually zero and one is used then we call this a binary classification problem for example spam or no spam on the other hand in regression we try to predict a continuous target value meaning the target value can have a more or less arbitrary value one example is to predict house prices based on given information about the house and the neighborhood the target variable which is the price can basically have any value here now that we know what supervised learning means let's have a look at concrete algorithms i will not explain them in detail here i simply name them so that you have heard of them they all have a unique design and can be different in the way how it stores the information mathematically how it solves the training process through mathematical operations and how it transforms the data this list is not exhaustive but here are 10 algorithms that are nice to know some of them can be used for either regression or classification and some can even be used for both cases popular algorithms are linear regression logistic regression decision trees random forest naive bayes perceptron and multi-layer perceptron support vector machines or short svm k-nearest neighbors or short knn adaboost and neural networks which are part of the deep learning field alright i hope you enjoyed
this video if you did so then please hit the like button and consider subscribing to the channel also if you want to try assembly ai for free then grab your free api token using the link in the description below and then i hope to
see you in the next video bye
"""
from dotenv import set_key
import streamlit as st
import time
import re
import pickle
import hashlib
import os
import json
import requests
from langchain_community.document_loaders import PyPDFLoader
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel
from typing import List
from tempfile import NamedTemporaryFile
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from youtube_transcript_api import YouTubeTranscriptApi
import os
import json
from pydantic import BaseModel
from typing import List
import requests
import pickle
import hashlib
import datetime
from datetime import datetime
import pandas as pd
 
 
# Set page configuration
st.set_page_config(page_title="Quizify", page_icon="🧠", layout="wide")
 
# Custom CSS for styling
custom_css = """
 
    <style>
 
   
    </style>
 
 
    """
# [data-testid="stSidebar"] { display: none;}
 
st.markdown(custom_css, unsafe_allow_html=True)
 
# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)
 
 
api_key = st.session_state.get('api_key', None)
endpoint = st.session_state.get('endpoint', None)
 
# Update file paths to include the Data folder
DATA_FOLDER = "Data"  # Specify your data folder here
 
QUIZ_PICKLE_FILE = os.path.join(DATA_FOLDER, "quiz_data.pkl")
USERS_PICKLE_FILE=os.path.join(DATA_FOLDER, "users.pkl")
 
#function for sidebar
def show_user_guide():
    """Display the user guide for Quizify."""
    st.header("Quizify User Guide")
    st.write("Welcome to Quizify! This guide will help you navigate the application, create quizzes, and track your progress,view quiz history and Maintain your profile")
   
    st.write("""
    - **Create Quizzes**: Upload PDF files or enter YouTube video links to generate quizzes automatically.
    - **Track Progress**: View your scores and history in your profile.
    - **User Authentication**: log in to save your quiz results.
    - **Save and Search**: You can save and search your quizes.
 
    ## User Authentication
    ### Login
    Enter your username and password to log in.
    Passwords must be at least 8 characters long and include uppercase letters, lowercase letters, numbers, and special characters.
 
    ## Dashboard Overview
    After logging in, you’ll be taken to your dashboard where you can:
    - Upload PDFs to generate quizzes.
    - Enter YouTube URLs for quiz generation.
    - View your quiz results and history.
 
    ## Creating Quizzes
    ### From PDF Files
    - **Upload a PDF**: Click the "Upload a PDF file" button and select a file from your device.
    - **Generate MCQs**: The application will extract text from the PDF and generate multiple-choice questions (MCQs) for you.
    - **Take the Quiz**: Answer the questions presented to you. You’ll receive immediate feedback on your answers.
 
    ### From YouTube Videos
    - **Enter YouTube URL**: In the "YouTube Link Quiz" tab, input the URL of the video you wish to use.
    - **Generate MCQs**: The app will fetch the transcript and create MCQs based on the video content.
 
    ## Taking Quizzes
    - **Answer Questions**: Select your answer from the provided options and submit.
    - **Feedback**: After submitting, you’ll see if your answer was correct or not.
    - **Navigate Questions**: Use "Next" to proceed to the next question or "Finish Quiz" to complete it. Your score will be displayed at the end by moving on the Quiz Result tab.
 
    ## Viewing Results
    - **Quiz Results**: After completing a quiz, you can view your score and detailed results, including the correct answers and your selected answers.
    - **User Profile**: Check your quiz history and see your highest scores and completion dates.
 
    ## Managing Your Account
    - **Logout**: Click the "Logout" button to safely exit your account.
 
    ## Troubleshooting
    If you encounter issues uploading files or generating quizzes, ensure that the PDF is not corrupted and that the YouTube URL is valid.
    For any technical issues, consider refreshing the page or checking your internet connection.
    """)
 
def save_user_results_login(username, quiz_title, score):
    try:
        user_results = load_user_results()  # Load existing results
        # Append new result for the user with a timestamp
        if username not in user_results:
            user_results[username] = []
        user_results[username].append((quiz_title, score, datetime.datetime.now().isoformat()))
    except Exception as e:
        print(f"Error saving user results: {e}")
   
 
# Function to load user quiz results from the pickle file
def load_user_results(username=None):
        try:
            if username:
                user_pickle_file = os.path.join(DATA_FOLDER, f"{username}_results.pkl")  # Updated path
                with open(user_pickle_file, 'rb') as f:
                    return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}
 
def load_users():
    """Load user data from the pickle file."""
    try:
        with open(USERS_PICKLE_FILE, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return {}
 
def save_users(users):
    """Save user data to the pickle file."""
    with open(USERS_PICKLE_FILE, 'wb') as f:
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
 
   
 
def save_user_results(username, quiz_title, score):
    # Create a filename for the user's pickle file (no directory, just the filename)
    user_pickle_file = os.path.join(DATA_FOLDER, f"{username}_results.pkl")  # Updated path
   
    # Load existing user results (if the file exists)
    if os.path.exists(user_pickle_file):
        with open(user_pickle_file, 'rb') as f:
            user_results = pickle.load(f)
    else:
        user_results = []  # Initialize an empty list if no results exist
   
    # Append the new quiz result (quiz title, score, and timestamp)
    user_results.append((quiz_title, score, datetime.now().isoformat()))
   
    # Save updated user results back to the pickle file
    with open(user_pickle_file, 'wb') as f:
        pickle.dump(user_results, f)
   
    print(f"User results saved for {username}.")
 
   
# Function to save MCQs to a pickle file
def save_quiz_to_pickle(mcqs, filename):
    with open(filename, 'wb') as f:
        pickle.dump(mcqs, f)
 
# Function to load MCQs from a pickle file
def load_quiz_from_pickle(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return []
 
headers = {
    'Content-Type': 'application/json',
    'api-key': api_key
}
 
 
def calculate_checksum(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()
 
def calculate_checksum_from_string(input_string):
    return hashlib.md5(input_string.encode('utf-8')).hexdigest()
class MCQ(BaseModel):
    question: str
    options: List[str]
    answer: str
    checksum: str  # Added for checksum tracking
 
class List_of_MCQs(BaseModel):
    mcqs: List[MCQ]
   
# Initialize session state
if "username" not in st.session_state:
    st.session_state.username = None
if "current_text" not in st.session_state:
    st.session_state.current_text = ""
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None
if "all_mcqs" not in st.session_state:
    st.session_state.all_mcqs = []
if "selected_answers" not in st.session_state:
    st.session_state.selected_answers = []
if "api_validation_status" not in st.session_state:
    st.session_state.api_validation_status = None  # Track API validation status
   
 
def split_text(text, max_length):
    """Split text into chunks of a maximum length."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]
 
def generate_mcqs_from_text(text):
    """Generate MCQs from text using OpenAI."""
    example_json = [
        {
            "question": "What is the capital of France?",
            "options": ["Paris", "London", "Berlin", "Madrid"],
            "answer": "Paris"
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "options": ["Earth", "Mars", "Jupiter", "Saturn"],
            "answer": "Mars"
        }
    ]
 
    text_chunks = split_text(text, 1000)
    all_mcqs = []
 
    for chunk in text_chunks:
        prompt = f"Generate multiple-choice questions (MCQs) from the following text in JSON format:\n\n{chunk}"
        messages = [
            {"role": "system", "content": f"Generate 2 MCQs from the following text in JSON format:\n{json.dumps(example_json, indent=4)}"},
            {"role": "user", "content": chunk}
        ]
        response = get_chat_completion(messages)
        try:
            mcqs = json.loads(response)
            if 'MCQs' in mcqs:
                all_mcqs.extend(mcqs['MCQs'])
            elif 'questions' in mcqs:
                all_mcqs.extend(mcqs['questions'])
        except KeyError as e:
            print(f"KeyError: {str(e)} - response did not have expected key.")
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {str(e)} - response could not be parsed.")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
        finally:
            continue
 
    return all_mcqs
def get_chat_completion(messages):
    data = {
        "messages": messages,
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    }
 
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))
    print("get response",response)
 
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None
 
# Main Application Logic
def main():
    with st.sidebar:
   # Add configuration section
        st.subheader("Configuration")
        api_key = st.text_input("Enter API Key:", type="password")
        endpoint = st.text_input("Enter API Endpoint:")
       
        if st.button("Save Configuration"):
                if api_key and endpoint:
                    # Save API Key and Endpoint to session state
                    st.session_state.api_key = api_key
                    st.session_state.endpoint = endpoint
                   
                    # Optionally save to .env file
                    set_key(".env", "AZURE_API_KEY", api_key)
                    set_key(".env", "API_ENDPOINT", endpoint)
                   
                    st.success("Configuration saved!")
                else:
                    st.error("Please enter both API Key and Endpoint.")
       
        show_user_guide()
 
 
    users = load_users()  # Load users from pickle file
 
    st.markdown("<h1 class='big-font'>Welcome to Quizify</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Embark on a journey of knowledge with AI-generated quizzes!</p>", unsafe_allow_html=True)
 
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
 
    if st.session_state.authenticated:
        show_dashboard()
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
                st.session_state.authenticated = True  
                st.session_state.username = username  
                st.rerun()  
            else:
                st.error("Invalid username or password.")
               
def handle_quiz_completion(username, quiz_title, score):
    # Save the user's score for the completed quiz
    #username=st.session_state.username
    save_user_results(username, quiz_title, score)
    st.success(f"Your score for {quiz_title} has been saved.")
 
 
api_key = st.session_state.get('api_key', None)
endpoint = st.session_state.get('endpoint', None)
 
def show_dashboard():
    st.success("Welcome back! You're now logged in.")
     # Add a logout button
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.clear()  # Clears all session state variables
        st.success("You have been logged out.")
        st.rerun()  # Rerun the app to show the login/signup form again
 
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["PDF to MCQs", "YouTube Link Quiz", "Quiz Results", "User Profile", "History"])
 
    with tab1:
        def validate_api_key(api_key, endpoint):
            if not api_key or not endpoint:
                print("API key or endpoint not found!")
                return False
 
            headers1 = {"Authorization": f"Bearer {api_key}"}  # Ensure you are passing the API key correctly
            try:
                print(f"Requesting {endpoint} with headers: {headers1}")  # Log the request details
                response = requests.get(endpoint, headers=headers1)
                print(f"Response Status Code: {response.status_code}")
                print(f"Response Body: {response.text}")  # Print response body for debugging
 
                if response.status_code == 200:
                    print("API key validated successfully.")
                    return True
                else:
                    print(f"API key validation failed. Status Code: {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Error during API validation: {e}")
                return False
 
 
        st.header("PDF to MCQs")
 
        AZURE_API_KEY = st.session_state.get('api_key', None)
        API_ENDPOINT = st.session_state.get('endpoint', None)
 
        # Uploaded file input
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
 
        # Check if there are existing MCQs and if uploaded file is provided
        if uploaded_file is not None:
            # Get the file name and set it as the quiz title
            quiz_title = uploaded_file.name.replace(".pdf", "")
            st.subheader(f"Quiz Title: {quiz_title}")
 
            file_bytes = uploaded_file.getvalue()
            new_checksum = calculate_checksum(file_bytes)
 
            # Load existing MCQs from pickle file
            all_mcqs = load_quiz_from_pickle(QUIZ_PICKLE_FILE)  # Implement this function to load from pickle file
            print(f"Loaded existing MCQs: {len(all_mcqs)} MCQs found.")
 
            # Check if MCQs for this PDF already exist
            existing_mcqs = [mcq for mcq in all_mcqs if mcq.get('checksum') == new_checksum]
            if existing_mcqs:
                st.warning("MCQs for this PDF already exist.")
                st.session_state.all_mcqs = existing_mcqs
                print("1\n")
 
                if 'quiz_ready_to_attempt' not in st.session_state:
                    # Show the toast message
                    st.toast("Quiz is ready to attempt")
                    # Set the session state variable to True
                    st.session_state.quiz_ready_to_attempt = True
 
            else:
                # If not, validate the Azure API key before processing
                if not AZURE_API_KEY:
                    st.error("API key not found! Please make sure it's set in sidebar")
                    quiz_title = None
                    st.session_state.api_validation_status = False  # Invalid API status
                    st.session_state.all_mcqs = []  # Reset to empty
                    print("API key not found.")
                else:
                    # Validate API key
                    if not validate_api_key(AZURE_API_KEY,API_ENDPOINT):  # Ensure you have this function defined
                        quiz_title = None
                        st.error("Invalid API key. Please provide a valid key.")
                        st.session_state.api_validation_status = False  # Invalid API status
                        st.session_state.all_mcqs = []  # Reset to empty
                    else:
                        st.session_state.api_validation_status = True  # API key is valid
 
                        # Proceed to save the uploaded file and generate new MCQs
                        with open("uploaded_file.pdf", "wb") as f:
                            f.write(file_bytes)
 
                        pdf_reader = PyPDFLoader("uploaded_file.pdf")
                        documents = pdf_reader.load()
                        pdf_text = "\n".join([doc.page_content for doc in documents])
                        os.remove("uploaded_file.pdf")
 
                        # Generate MCQs from the PDF text
                        new_mcqs = generate_mcqs_from_text(pdf_text)
                        if new_mcqs:  # Only proceed if new MCQs were generated
                            for mcq in new_mcqs:
                                mcq['checksum'] = new_checksum
 
                            # Append new MCQs to the existing list and save
                            all_mcqs.extend(new_mcqs)
                            st.session_state.all_mcqs = all_mcqs  # Update session state with all MCQs
 
                            save_quiz_to_pickle(all_mcqs, QUIZ_PICKLE_FILE)
                            st.success("Generated and saved MCQs.")
                           
                        else:
                            st.error("No MCQs were generated from the uploaded PDF.")                          
        # Quiz functionality
        if st.session_state.all_mcqs:
            all_mcqs = st.session_state.all_mcqs
            current_question = st.session_state.current_question
 
            if current_question < len(all_mcqs):
                mcq = all_mcqs[current_question]
 
                st.write(f"**Question {current_question + 1}:** {mcq['question']}")
                selected_option = st.radio("Select an option:", mcq['options'], key=f"question_{current_question}")
 
                if st.button("Submit") and not st.session_state.show_feedback:
                    st.session_state.selected_option = selected_option
                    st.session_state.selected_answers.append(selected_option)  # Store the selected answer
                    st.session_state.show_feedback = True
 
                    # Check the answer and update score only once per question
                    if st.session_state.selected_option == mcq['answer']:
                        st.success("Correct!")
                        st.session_state.score += 1
                    else:
                        st.error(f"Wrong! The correct answer is: {mcq['answer']}")
 
                # Show "Next" button only after feedback is shown
                if st.session_state.show_feedback and current_question < len(all_mcqs) - 1:
                    if st.button("Next"):
                        st.session_state.current_question += 1
                        st.session_state.show_feedback = False
                        st.session_state.selected_option = None
                        st.rerun()
 
                # Show "Finish Quiz" button when on the last question
                if st.session_state.show_feedback and current_question == len(all_mcqs) - 1:
                    if st.button("Finish Quiz!! Click on Quiz Results Tab to view quiz result"):
                        # Check if there are any MCQs before completing the quiz
                        if len(st.session_state.all_mcqs) > 0:
                            score = st.session_state.score
                            st.session_state.current_question += 1
                            st.session_state.show_feedback = False
                            st.session_state.selected_option = None    
                            # Call the completion handler only if there are MCQs
                            handle_quiz_completion(st.session_state.username, quiz_title, score)
                            st.rerun()
                           
        else:
            st.write("Please upload a PDF file to generate MCQs.")
            print("No MCQs available. Prompting user to upload a PDF.")
   
    with tab2:
        st.header("YouTube Link Quiz")
        youtube_url = st.text_input("Enter YouTube video URL:")
   
        if youtube_url:
            video_checksum = calculate_checksum_from_string(youtube_url)
            all_mcqs = load_quiz_from_pickle(QUIZ_PICKLE_FILE)
   
            existing_mcqs = [mcq for mcq in all_mcqs if mcq.get('checksum') == video_checksum]
            if existing_mcqs:
                st.warning("MCQs for this YouTube video already exist.")
                st.session_state.all_mcqs = existing_mcqs
            else:
                transcript_text = get_transcript(youtube_url)
                if transcript_text:
                    st.session_state.current_text = transcript_text
                    new_mcqs = generate_mcqs_from_text(transcript_text)
                    for mcq in new_mcqs:
                        mcq['checksum'] = video_checksum
   
                    all_mcqs.extend(new_mcqs)
                    st.session_state.all_mcqs = new_mcqs
                    save_quiz_to_pickle(all_mcqs, QUIZ_PICKLE_FILE)
                    st.success("Generated and saved MCQs for the YouTube video.")
    with tab3:
        st.header("Quiz Results")
        st.write("This Tab will show the result of your last attempted Quiz with a complete overview.")
 
        # Check if the quiz has been completed and if valid MCQs are present
        if 'all_mcqs' in st.session_state and st.session_state.all_mcqs and st.session_state.current_question >= len(st.session_state.all_mcqs):
            score = st.session_state.get('score', 0)
            total_questions = len(st.session_state.all_mcqs)
            st.write(f"Your score: {score} out of {total_questions}")
 
            # Show full quiz results in a table
            results_data = []
            for idx, mcq in enumerate(st.session_state.all_mcqs):
                selected_answer = st.session_state.selected_answers[idx] if idx < len(st.session_state.selected_answers) else "Not answered"
                correct_answer = mcq['answer']
 
                # Highlight incorrect answers in red and correct answers in green
                if selected_answer == correct_answer:
                    results_data.append({
                        "Question": f"{idx + 1}: {mcq['question']}",
                        "Your Answer": f'<div style="background-color: #d4edda; color: black;">{selected_answer}</div>',  # Light green
                        "Correct Answer": correct_answer
                    })
                else:
                    results_data.append({
                        "Question": f"{idx + 1}: {mcq['question']}",
                        "Your Answer": f'<div style="background-color: #f8d7da; color: black;">{selected_answer}</div>',  # Light red
                        "Correct Answer": correct_answer
                    })
 
            # Create a DataFrame
            results_df = pd.DataFrame(results_data)
 
            # Display the results as a DataFrame with custom HTML rendering
            st.markdown(results_df.to_html(escape=False), unsafe_allow_html=True)
 
            # Save the results for the user only if MCQs were generated
            username = st.session_state.get('username')
            if 'uploaded_file' in st.session_state:
                quiz_title = st.session_state.uploaded_file.name.replace(".pdf", "")
                save_user_results_login(username, quiz_title, score)  # Save results
                st.success(f"Results saved for {username} (Quiz: {quiz_title}).")
            else:
                print("No PDF file found. Please upload a file to save results.")
        else:
            st.write("Please complete the quiz to see your results.")
 
 
 
    with tab4:
       
        def display_unique_quiz_results(user_results):
            unique_quizzes = {}
 
            # Loop through user results and collect the highest score for each unique quiz title
            for quiz_title, score, timestamp in user_results:
                if quiz_title not in unique_quizzes:
                    unique_quizzes[quiz_title] = (score, timestamp)
                else:
                    # Store the highest score if desired
                    if score > unique_quizzes[quiz_title][0]:
                        unique_quizzes[quiz_title] = (score, timestamp)
           
            profile_data = []
            for quiz_title, (score, timestamp) in unique_quizzes.items():
                dt = datetime.fromisoformat(timestamp)  # Convert the timestamp to a datetime object
                profile_data.append({
                    "Quiz Title": quiz_title,
                    "Score": score,
                    "Date": dt.date().strftime('%d-%m-%Y'),  # Format date
                    "Time": dt.time().strftime('%I:%M %p')   # Format time in 12-hour format
 
                })
 
            # Convert the profile data into a pandas DataFrame and display it
            if profile_data:  # Only display if there are results
                df = pd.DataFrame(profile_data)
                st.write(df)  # Display user results as a DataFrame
            else:
                st.write("No quiz results available.")
 
        st.header("User Profile")
 
        username = st.session_state.get('username')
        user_results = load_user_results(username)
 
        if user_results:
            st.write(f"Welcome, {username}!")
            display_unique_quiz_results(user_results)
        else:
            st.write("No quiz results available.")
 
    with tab5:
        st.header("History")
        st.write("This Tab will show all of your Quizzes attempts with your scores in them")
       
        # Local History
        st.subheader("Local History (Your Quizzes)")
        username = st.session_state.get('username')
        user_results = load_user_results(username)
 
        if user_results:
            # Sort user_results in descending order based on the timestamp
            user_results.sort(key=lambda x: x[2], reverse=True)  # x[2] is the timestamp
           
            history_data = []
            for quiz_title, score, timestamp in user_results:
                dt = datetime.fromisoformat(timestamp)  # Convert the timestamp to a datetime object
                history_data.append({
                    "Quiz Title": quiz_title,
                    "Score": score,
                    "Date": dt.date().strftime('%Y-%m-%d'),  # Format date
                    "Time": dt.time().strftime('%I:%M %p')   # Format time in 12-hour format
 
                })
           
            st.write(pd.DataFrame(history_data))  # Display local history as a DataFrame
        else:
            st.write("No local quiz results available.")
   
def get_transcript(youtube_url):
    """Extract transcript from a YouTube video."""
    video_id = extract_video_id(youtube_url)
    if video_id:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry['text'] for entry in transcript])
            return transcript_text
        except Exception as e:
            st.error(f"Error fetching transcript: {e}")
            st.info("Using test text instead.")
            return ""
    else:
        st.error("Invalid YouTube URL.")
        return ""
 
 
def extract_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    if "youtube.com/watch?v=" in url:
        return url.split("youtube.com/watch?v=")[-1]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1]
    return None
 
 
if __name__ == "__main__":
    main()
 
