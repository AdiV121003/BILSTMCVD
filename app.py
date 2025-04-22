import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
from pygments.lexers import CLexer
from pygments import lex
from keras.preprocessing.sequence import pad_sequences

# Load model and vocabulary
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.h5", compile=False)

@st.cache_data
def load_vocab():
    with open("vocab.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
vocab = load_vocab()

# Constants
MAX_SEQ_LENGTH = 1429
THRESHOLD = 0.5 # Adjust as needed

# Preprocessing Functions
def preprocess_code(code):
    """Removes comments, normalizes numbers, and tokenizes C code."""
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    #code = re.sub(r'".*?"', '', code)
    code = re.sub(r'\b[-+]?\d*\.?\d+\b', 'NUMBER', code)
    return code.strip()

def clexer_tokenize(code):
    """Tokenizes C code using Pygments."""
    lexer = CLexer()
    return [token[1] for token in lex(preprocess_code(code), lexer)]

def tokens_to_integers(tokens):
    """Converts tokens to integer values based on vocabulary."""
    return [vocab.get(token, 0) for token in tokens]

def preprocess(func_before):
    """Tokenizes, converts to integers, and pads input."""
    tokens = clexer_tokenize(func_before)
    token_ids = tokens_to_integers(tokens)
    padded_sequence = pad_sequences([token_ids], maxlen=MAX_SEQ_LENGTH, padding="post")
    return padded_sequence

# Streamlit UI
st.title("🔍 Vulnerability Detection in C Code")
st.write("Enter a C function to check if it's vulnerable.")

func_before = st.text_area("Paste your C function here:", height=200)

if st.button("Check Vulnerability"):
    if func_before.strip() == "":
        st.error("⚠️ Please enter a valid C function.")
    else:
        processed_input = preprocess(func_before)
        st.write(f"Raw Model Output (confidence): {raw_prediction[0][0]:.4f}")
        print("Raw Prediction Output:", raw_prediction)
        binary_prediction = (raw_prediction > 0.5).astype("int32")  # Convert to 0 or 1

        # Extract float value
        confidence_score = float(raw_prediction[0][0])

        result = "🔴 Vulnerable" if binary_prediction[0][0] == 1 else "🟢 Not Vulnerable"

        st.write(f"### **Prediction: {result}**")

