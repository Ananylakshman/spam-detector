import streamlit as st
import joblib
import re
import string

# Load your saved model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
def preprocess_message(msg):
    msg = msg.lower()
    msg = re.sub(r'\d+', '', msg)  # remove digits
    msg = msg.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return msg

st.title("Spam Message Detector")

# Input text box
message = st.text_area("Enter your message here:")

if st.button("Check Spam"):
    if message.strip() == "":
        st.warning("Please enter a message to check!")
    else:
        clean_msg = preprocess_message(message)
        vec_msg = vectorizer.transform([clean_msg])
        prediction = model.predict(vec_msg)[0]
        label = "Spam" if prediction == 1 else "Not Spam"
        st.success(f"The message is predicted to be: {label}")
