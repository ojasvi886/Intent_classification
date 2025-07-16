import streamlit as st
import pickle

with open("logistic_model.pkl", "rb") as f1:
    model = pickle.load(f1)
with open("tfidf_vectorizer.pkl", "rb") as f2:
    vectorizer = pickle.load(f2)

st.title("Intent Detection with Logistic Regression")
st.markdown("Enter a user query, and the model will predict the intent behind it.")
user_input = st.text_area("Enter your sentence here:", height=100)

if st.button("Predict Intent"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence to classify.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        st.success(f"Predicted Intent: **{prediction}**")
