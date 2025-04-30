import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the image sentiment model
emoji_sentiment_model = load_model('emoji_sentiment_model.h5')

# Load the TF-IDF vectorizer and Logistic Regression model for text
with open('sentiment_model66.pkl', 'rb') as model_file:
    text_sentiment_model = pickle.load(model_file)

with open('tfidf_vectorizer66.pkl', 'rb') as vec_file:
    tfidf_vectorizer = pickle.load(vec_file)

# Text prediction function (using sklearn model and tfidf)
def predict_text_sentiment(text):
    processed_text = " ".join(word for word in text.split())  # Basic cleaning; you can enhance this
    vectorized = tfidf_vectorizer.transform([processed_text]).toarray()
    prediction = text_sentiment_model.predict(vectorized)
    return "Positive Sentiment" if prediction[0] == 1 else "Negative Sentiment"

# Image prediction function remains the same
def predict_image_sentiment(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = emoji_sentiment_model.predict(img_array)
    return "Positive Sentiment" if prediction > 0.5 else "Negative Sentiment"

# Streamlit UI
st.title("Sentiment Analysis App")

# Text Review Prediction
st.subheader("Text Review Sentiment Prediction")
review_text = st.text_area("Enter your review text:")
if st.button("Predict Text Sentiment"):
    if review_text.strip():
        sentiment = predict_text_sentiment(review_text)
        st.write(f"**Sentiment of the text:** {sentiment}")
    else:
        st.warning("Please enter a review text.")

# Image Sentiment Prediction
st.subheader("Image Sentiment Prediction")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if st.button("Predict Image Sentiment") and uploaded_image:
    image_path = uploaded_image.name
    with open(image_path, 'wb') as f:
        f.write(uploaded_image.getbuffer())
    sentiment = predict_image_sentiment(image_path)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Sentiment of the uploaded image:** {sentiment}")

# Combined Prediction
st.subheader("Combined Sentiment Prediction (Text + Image)")
combined_text = st.text_area("Enter review text for combined prediction:")
uploaded_image_for_combined = st.file_uploader("Upload an image for combined prediction", type=["jpg", "jpeg", "png"], key="combined_image")

if st.button("Predict Combined Sentiment"):
    if combined_text.strip() and uploaded_image_for_combined:
        # Text prediction
        text_sentiment = predict_text_sentiment(combined_text)

        # Image prediction
        image_path = uploaded_image_for_combined.name
        with open(image_path, 'wb') as f:
            f.write(uploaded_image_for_combined.getbuffer())
        image_sentiment = predict_image_sentiment(image_path)

        # Display uploaded image
        st.image(uploaded_image_for_combined, caption="Uploaded Image for Combined Prediction", use_column_width=True)

        # Combine predictions
        if text_sentiment == "Positive Sentiment" and image_sentiment == "Positive Sentiment":
            final_sentiment = "Positive Sentiment"
        elif text_sentiment == "Negative Sentiment" and image_sentiment == "Negative Sentiment":
            final_sentiment = "Negative Sentiment"
        else:
            final_sentiment = "Neutral sentiment"

        st.write(f"**Text Sentiment:** {text_sentiment}")
        st.write(f"**Image Sentiment:** {image_sentiment}")
        st.write(f"**Combined Sentiment:** {final_sentiment}")
    else:
        st.warning("Please provide both text and image for combined sentiment prediction.")
