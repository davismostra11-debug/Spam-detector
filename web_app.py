
import streamlit as st
import joblib

# Page settings
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Load existing trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("spam_vectorizer.pkl")

# App heading
st.title("📩 Spam Message Detector")
st.write("Check whether a message is Spam or Not Spam.")

st.divider()

# Message input
message = st.text_area(
    "Enter your message below:",
    placeholder="Type or paste your message here..."
)

# Prediction button
if st.button("Check Message", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        # Convert message using the saved vectorizer
        message_vectorized = vectorizer.transform([message])

        # Predict using the existing trained model
        prediction = model.predict(message_vectorized)[0]

        st.subheader("Prediction Result")

        if str(prediction).lower() in ["spam", "1"]:
            st.error("🚨 This message is SPAM!")

        else:
            st.success("✅ This message is NOT SPAM!")

st.divider()
st.caption("Spam Detection | Machine Learning Project")