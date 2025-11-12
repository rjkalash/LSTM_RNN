# Load all the necessary libraries
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained model
model = load_model('lstm_text_generator.h5')

# Function to generate text
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Streamlit app
st.title("LSTM Text Generator")
seed_text = st.text_input("Enter Seed Text", "To be or not to be") 
# predict the word
if st.button("Predict"):
    max_sequence_len = model.input_shape[1]+1
    next_words = generate_text(seed_text=seed_text, next_words=1, max_sequence_len=max_sequence_len )
    st.write(next_words)
   

