# -*- coding: utf-8 -*-
"""BILSTM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1H9wy3AxnAWwAcuqOwoX_OiZyhUlex66f
"""

!git clone https://github.com/OscarMC28/Software-Vulnerability-BigVul.git

!git clone https://github.com/OscarMC28/Software-Vulnerability-BigVul.git

!pip install datasets

import pandas as pd
import numpy as np
import seaborn as sns
import os
import datasets

# Extend number of columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Open file from Google Drive
from google.colab import drive
drive.mount('/content/drive')

from datasets import load_dataset

ds = load_dataset("bstee615/bigvul")

# Access the 'train' split of the DatasetDict and convert to pandas DataFrame
df = ds['train'].to_pandas() # Access the 'train' split before calling to_pandas()

# Display first few rows
print(df.head())

# Display first few rows
print(df.head())

import matplotlib.pyplot as plt

# Plot the distribution of 'lang' feature
plt.figure(figsize=(8, 6))
sns.countplot(data=df, y='lang', palette='pastel', order=df['lang'].value_counts().index)
plt.title('Distribution of Programming Languages')
plt.ylabel('Language')
plt.xlabel('Count')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = {
    'vul': [0, 1, 0, 1, 0],
    'lang': ['C', 'C', 'CPP', 'CPP', 'C++'],
    'count': [175211, 10786, 2522, 114, 3]
}
df_lang = pd.DataFrame(data)

# Getting the value counts
counts = df_lang.groupby(['lang', 'vul']).sum()['count']

# Unstacking the Series to get a DataFrame suitable for bar plot
unstacked_counts = counts.unstack().fillna(0)

fig, ax = plt.subplots()

# Setting the positions and width for the bars
width = 0.35  # Width of the bars
ind = range(len(unstacked_counts))  # Number of groups (languages)

# Plotting each category
p1 = ax.bar(ind, unstacked_counts[0], width, label='Vul 0')
p2 = ax.bar([i + width for i in ind], unstacked_counts[1], width, label='Vul 1')

# Setting labels, title and axes ticks
ax.set_xlabel('Language')
ax.set_ylabel('Number of Vulnerabilities')
ax.set_title('Vulnerabilities by Programming Language')
ax.set_xticks([i + width / 2 for i in ind])
ax.set_xticklabels(unstacked_counts.index)
ax.legend()

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
# take 6 variables only related to the code itself
# Check available columns in df
print(df.columns)

# Select existing columns or adjust column names
df2 = df[['lang', 'func_before', 'func_after', 'vul']]  # Example: selecting only existing columns

# If the desired columns exist with different names, update the column names accordingly.
# For example, if 'lines_before' is actually named 'lines_before_fix':
# df2 = df[['lang', 'func_before', 'func_after', 'lines_before_fix', 'lines_after_fix', 'vul', 'vul_func_with_fix']]

# Filter the dataframe to only include rows with C
df3 = df2[df2['lang'] == "C"]

import warnings
warnings.filterwarnings("ignore")

# Distribution of vul
V_dist = sns.displot(df3['vul']) # Use df3 instead of df4
plt.show()

# Save the df3 dataframe
from google.colab import drive
drive.mount('/content/drive')

# Save the df6 dataframe to a CSV file in Google Drive for saving memory
save_path = "/content/drive/MyDrive/df_preprocessed.csv"
df3.to_csv(save_path, index=False)

!pip install tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import os

# Load the data: Open file from Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define the file path in Google Colab after mounting the drive
file_path = "/content/drive/My Drive/df_preprocessed.csv"

# Read the CSV file into a DataFrame
try:
    df1 = pd.read_csv(file_path, na_filter=False)

except FileNotFoundError:
    print("The CSV file does not exist.")

# Reset index
df = df1.reset_index(drop=True)
print("shape:", df.shape)
df.head(5)

# Split the data
np.random.seed(42) #Set a random seed for reproducibility
Y = np.array(df["vul"]) # Define the target/response feature and convert it into an array for easy manipulation
df = df["func_before"].tolist() # select feature function code before fixing and convert into list
Y[Y > 0] = 1 # Set the binary classification (0=non-vulnerable code and 1=vulnerable code)

# Split randomly the data
num_samples = len(df)
train_samples = int(round(num_samples * 0.60))
val_samples = int(round(num_samples * 0.20))
samples = np.random.choice(len(Y), num_samples, replace=False)

X_train = [df[i] for i in samples[:train_samples]]
y_train = Y[samples[:train_samples]]
X_val = [df[i] for i in samples[train_samples:train_samples+val_samples]]
y_val = Y[samples[train_samples:train_samples+val_samples]]
X_test = [df[i] for i in samples[train_samples+val_samples:]]
y_test = Y[samples[train_samples+val_samples:]]

import matplotlib.pyplot as plt

# Compute number of samples for each set
num_train = len(X_train)
num_val = len(X_val)
num_test = len(X_test)

# Create a bar chart
labels = ['Training set', 'Validation set', 'Test set']
counts = [num_train, num_val, num_test]

bars = plt.bar(labels, counts, color=['blue', 'green', 'red'])
plt.ylabel('Number of Samples')
plt.title('Distribution of Data')

# Add values over each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, str(height),
             ha='center', va='bottom')

plt.show()

# Check the values in each variable: sanity check
print("X_train:", len(X_train))
print("y_train:", len(y_train))
print("X_val:", len(X_val))
print("y_val:", len(y_val))
print("X_test:", len(X_test))
print("y_test:", len(y_test))

# Calculate class distribution per each dataset
unique, counts = np.unique(y_train, return_counts=True)
class_counts_y_train = dict(zip(unique, counts))

unique, counts = np.unique(y_val, return_counts=True)
class_counts_y_val = dict(zip(unique, counts))

unique, counts = np.unique(y_test, return_counts=True)
class_counts_y_test = dict(zip(unique, counts))

print("class_counts_y_train:", class_counts_y_train)
print("class_counts_y_val:", class_counts_y_val)
print("class_counts_y_test:", class_counts_y_test)

from collections import defaultdict, Counter
from pygments.lexers import CLexer
from pygments import lex
from tqdm import tqdm
import re

# Define preprocessing and tokenization
def preprocess_code(code):
    code = re.sub(r'//.*', '', code)  # remove single-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # remove multi-line comments
    code = re.sub(r'".*?"', '', code)  # remove string literals
    code = re.sub(r'\b[-+]?\d*\.?\d+\b', 'NUMBER', code)  # normalize numbers
    return code.strip()

# Lexex tokeniser for C programing
def clexer_tokenize(code):
    lexer = CLexer()
    return [token[1] for token in lex(preprocess_code(code), lexer)]

# Tokenize training data
X_train_tokenized = [clexer_tokenize(code) for code in tqdm(X_train, desc="Tokenizing Train Data")]

# Calculate Token Frequencies
token_freq = Counter(token for tokens in X_train_tokenized for token in tokens)

# Print the total number of tokens
print("Total number of tokens:", len(token_freq))

import matplotlib.pyplot as plt

# Determine a frequency threshold from percentiles
tokens, frequencies = zip(*sorted(token_freq.items(), key=lambda x: x[1], reverse=True))
percentiles = [75, 90, 95, 97, 99]
percentile_values = np.percentile(frequencies, percentiles)

# Plot the token frequencies
plt.figure(figsize=(10,6))
plt.hist(frequencies, bins=100, edgecolor='black', alpha=0.7, log=True)  # log scale for better visualization
plt.title('Token Frequency Distribution')
plt.xlabel('Frequency')
plt.ylabel('Number of Tokens (log scale)')

# Plot the percentiles on the histogram
for perc, value in zip(percentiles, percentile_values):
    plt.axvline(value, color='red', linestyle='dashed', linewidth=1, label=f'{perc}th Percentile')

# Print the determined threshold frequencies and count tokens above and below each threshold
for perc, value in zip(percentiles, percentile_values):
    # Get the number of tokens above the current percentile value
    num_tokens_above_value = sum(i >= value for i in frequencies)
    # Get the number of tokens below the current percentile value
    num_tokens_below_value = sum(i < value for i in frequencies)
    print(f"{perc}th percentile frequency: {value} (Number of tokens above this value: {num_tokens_above_value}, Number of tokens below or equal to this value: {num_tokens_below_value})")

plt.legend()
plt.tight_layout()
plt.show()

# Choose a threshold value (95th percentile)
threshold_frequency = percentile_values[2]

# Filter out tokens that are below the threshold
frequent_tokens = {token for token, freq in token_freq.items() if freq >= threshold_frequency}
print("Number of frequent tokens:", len(frequent_tokens))

# Filter the tokenized data based on the threshold
X_train_filtered = [[token for token in tokens if token in frequent_tokens] for tokens in X_train_tokenized]
print("Example of filtered data:", X_train_filtered[0])

# Use frequent tokens for filtering
frequent_tokens = set(token for token, freq in token_freq.items() if freq >= threshold_frequency)

#Create a vocabulary with the frequent tokens
vocab = {token: idx+1 for idx, token in enumerate(frequent_tokens)}  # Adding 1 in the index because 0 is reserved for padding
vocab_size = len(vocab)

print("vocab_size:", vocab_size)

#from collections import defaultdict
#from operator import itemgetter
from keras.preprocessing.sequence import pad_sequences
from pygments.lexers import CLexer
from pygments import lex
from tqdm import tqdm
import re


# Function to tokenise dataset with frequent tokens/vocabulary
def frequent_tokens_filter(code):
    return [token for token in clexer_tokenize(code) if token in frequent_tokens]

# Tokenize using the frequent tokens
X_train_tokenized = [frequent_tokens_filter(code) for code in tqdm(X_train, desc="Tokenizing Train Data with Frequent Tokens")]
X_val_tokenized = [frequent_tokens_filter(code) for code in tqdm(X_val, desc="Tokenizing Validation Data with Frequent Tokens")]
X_test_tokenized = [frequent_tokens_filter(code) for code in tqdm(X_test, desc="Tokenizing Test Data with Frequent Tokens")]

# Function to convert tokens to integers
def tokens_to_integers(tokens):
    return [vocab[token] for token in tokens if token in vocab]

# Convert tokens to integers
X_train_seq = [tokens_to_integers(tokens) for tokens in X_train_tokenized]
X_val_seq = [tokens_to_integers(tokens) for tokens in X_val_tokenized]
X_test_seq = [tokens_to_integers(tokens) for tokens in X_test_tokenized]

import matplotlib.pyplot as plt

# Calculate sequence lengths
sequence_lengths = [len(seq) for seq in X_train_seq + X_val_seq + X_test_seq]

# Calculate percentiles
percentiles = [np.percentile(sequence_lengths, p) for p in [75, 95, 98, 99]]

# Plotting the distribution
plt.figure(figsize=(10,6))
plt.hist(sequence_lengths, bins=50, edgecolor='black', alpha=0.7)
plt.title('Sequence Length Distribution')
plt.xlabel('Length of Sequences')
plt.ylabel('Number of Sequences')

# Plotting the percentiles
for p in percentiles:
    plt.axvline(p, color='red', linestyle='dashed', linewidth=1)

# Display the plot with legend
labels = ['75th', '95th', '98th', '95th']
plt.legend(['Sequence Lengths'] + labels)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Calculate percentiles values
percentile_values = [75, 95, 98, 99, 100]
percentiles = [np.percentile(sequence_lengths, p) for p in percentile_values]

for p_val, p_result in zip(percentile_values, percentiles):
    print(f"The {p_val}th percentile is: {p_result}")

# Padding and truncating sequences with 98th percentile length of sequences
max_length = int(np.percentile([len(seq) for seq in X_train_seq + X_val_seq + X_test_seq], 98))
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post' )
X_val_padded = pad_sequences(X_val_seq, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Vector representation: set dimension embedding
embedding_dim = 100

!pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_padded, y_train)

import numpy as np
import matplotlib.pyplot as plt

# Calculate class distribution
unique, counts = np.unique(y_train_resampled, return_counts=True)
class_counts = dict(zip(unique, counts))

# Plotting the distribution
plt.figure(figsize=(8,6))
bars = plt.bar(class_counts.keys(), class_counts.values(), edgecolor='black', alpha=0.7)
plt.title('Training Class Distribution: Oversampling')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(list(class_counts.keys()))  # Ensure all classes are displayed on the x-axis
plt.grid(axis='y')

# Annotate the values over the bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom')

plt.tight_layout()

# Display the plot
plt.show()

from time import process_time
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Evaluation of Model

def plot_training(history):
    fig = plt.figure(figsize=[20, 5])
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history['loss'], label="Training Loss")
    ax.plot(history.history['val_loss'], label="Validation Loss")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss vs Validation Loss')
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history['accuracy'], label="Train Accuracy")
    ax.plot(history.history['val_accuracy'], label="Val Accuracy")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy vs Validation Accuracy')
    ax.legend()

    plt.show()


def eval_model(model, X_train, y_train, X_test, y_test, train_time=None):
    """
    Evaluates the model and prints/visualizes various performance metrics.

    Arguments:
    - model: Keras model that needs to be evaluated.
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Testing data and labels.
    - train_time: Optional, time taken to train the model.

    Returns:
    None
    """

    # Evaluate on the test set
    test_scores = model.evaluate(X_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])
    print("")

    fig, axes = plt.subplots(1, 2, figsize=[20, 8])

    # Predict on the training set and calculate the inference time
    inference_start = process_time()
    train_pred = (model.predict(X_train) > 0.5).astype("int32")
    inference_end = process_time()

    # Compute and visualize confusion matrix for training data
    confusion_mtx_train = confusion_matrix(y_train, train_pred)
    sns.heatmap(confusion_mtx_train, annot=True, fmt='g', ax=axes[0])
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_title(f'Training, F1 Score: {f1_score(y_train, train_pred):.4f}')
    print(classification_report(y_train, train_pred))

    # Predict on the test set and calculate the inference time
    pred_start = process_time()
    test_pred = (model.predict(X_test) > 0.5).astype("int32")
    pred_end = process_time()

    # Compute and visualize confusion matrix for test data
    confusion_mtx_test = confusion_matrix(y_test, test_pred)
    sns.heatmap(confusion_mtx_test, annot=True, fmt='g', ax=axes[1])
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].set_title(f'Testing, F1 Score: {f1_score(y_test, test_pred):.4f}')

    if train_time:
        print(f'Training Time: {train_time:.4f} seconds')

    print(f'Inference Time (training set): {inference_end - inference_start:.4f} seconds')
    print(f'Inference Time (testing set): {pred_end - pred_start:.4f} seconds\n')
    print(classification_report(y_test, test_pred))

    plt.tight_layout()
    plt.show()


# allow you to stop model training after a set period of time
class TrainForTime(keras.callbacks.Callback):
    def __init__(self, train_time_mins=5,):
        super().__init__()

        self.train_time_mins = train_time_mins
        self.epochs = 0
        self.train_time = 0
        self.end_early = False

    def on_train_begin(self, logs=None):
        # save the start time
        self.start_time = tf.timestamp()

    def on_epoch_end(self, epoch, logs=None):
        self.epochs += 1
        current_time = tf.timestamp()
        training_time = (current_time - self.start_time)
        if (training_time / 60) > self.train_time_mins:
            self.train_time = current_time - self.start_time
            self.model.stop_training = True
            self.end_early = True

    def on_train_end(self, logs=None):
        if self.end_early:
            print('training time exceeded and ending early')
            print(f'training ended on epoch {self.epochs}')
            print(f'training time = {self.train_time / 60} mins')

# Save the best model
from keras.callbacks import ModelCheckpoint
checkpoint_filepath = "/content/drive/MyDrive/Colab Notebooks/best_model_{epoch:02d}-{val_loss:.2f}.h5"
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                   save_best_only=True,  # Only save a model if 'val_loss' has improved.
                                   monitor='val_loss',
                                   mode='min',  # 'min' mode means the callback saves when 'val_loss' is minimized.
                                   verbose=1)

import time
def track_time(start_time):
    return time.time() - start_time

# Function to evaluate the traditional ML models
def eval_model_tml(model, X_train, Y_train, X_test, Y_test):
    fig = plt.figure(figsize=[15, 6])

    ax = fig.add_subplot(1, 2, 1)
    conf = ConfusionMatrixDisplay.from_estimator(model, X_train, Y_train, normalize='true', ax=ax)
    pred = model.predict(X_train)
    conf.ax_.set_title('Training Set Performance: ' + str(sum(pred == Y_train)/len(Y_train)))

    ax = fig.add_subplot(1, 2, 2)
    conf = ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test, normalize='true', ax=ax)
    pred = model.predict(X_test)
    conf.ax_.set_title('Test Set Performance: ' + str(sum(pred == Y_test)/len(Y_test)))

    plt.show()


import matplotlib.pyplot as plt

def plot_training_histories(histories):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    for history in histories:
        ax1.plot(history.history['loss'], label="Training Loss")
        ax1.plot(history.history['val_loss'], label="Validation Loss")
        ax2.plot(history.history['accuracy'], label="Train Accuracy")
        ax2.plot(history.history['val_accuracy'], label="Val Accuracy")

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss vs Validation Loss')
    ax1.legend()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy vs Validation Accuracy')
    ax2.legend()

    plt.show()


import pickle
import matplotlib.pyplot as plt

def save_history_to_file(history, filename):
    with open(filename, 'wb') as file:
        pickle.dump(history.history, file)

def load_history_from_file(filename):
    with open(filename, 'rb') as file:
        history = pickle.load(file)
    return history


def save_metrics_to_file(metrics, filename):
    with open(filename, 'wb') as file:
        pickle.dump(metrics, file)

def load_metrics_from_file(filename):
    with open(filename, 'rb') as file:
        metrics = pickle.load(file)
    return metrics

import tensorflow as tf
from tensorflow.keras import backend as K

class Attention(tf.keras.layers.Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = tf.keras.initializers.get('glorot_uniform')

        self.W_regularizer = W_regularizer
        self.b_regularizer = b_regularizer
        self.W_constraint = W_constraint
        self.b_constraint = b_constraint

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D

# Class weight computation for class imbalance
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

# Optimizer with learning rate
optimizer = Adam(learning_rate=0.001)

# Model definition
model_BiLSTM = Sequential()
# Add an embedding layer/vocab_size/max_length
model_BiLSTM.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_length, trainable=True))
# Add a Bidirectional LSTM layer with dropout
model_BiLSTM.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.25)))
# Add a Global MaxPooling layer
model_BiLSTM.add(GlobalMaxPooling1D())
# Add an additional Dense layer with ReLU activation
model_BiLSTM.add(Dense(64, activation='relu'))
# Add a Dropout layer
model_BiLSTM.add(Dropout(0.5))
# Add the output Dense layer with sigmoid activation for binary classification
model_BiLSTM.add(Dense(1, activation='sigmoid'))
# Compile the model with optimizer, loss, and metrics
model_BiLSTM.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model_BiLSTM.build(input_shape=(None, max_length)) # Build the model explicitly

print(model_BiLSTM.summary())
plot_model(model_BiLSTM, show_shapes=True)

import time
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

# Training phase
def track_time(start_time):
    """Calculate the elapsed time in seconds given a starting time"""
    return time.time() - start_time

def calculate_accuracy(y_true, y_pred_probs):
    """Calculate accuracy given true labels and predicted probabilities"""
    y_pred = (y_pred_probs > 0.5).astype("int32")
    return accuracy_score(y_true, y_pred)

# 1. Training the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
train_time_mins = 90 # early time
train_time_callback = TrainForTime(train_time_mins=train_time_mins) # early time
class_weights = class_weight_dict

train_start = time.time()

history_BiLSTM = model_BiLSTM.fit(
    X_train_padded, y_train,
    validation_data=(X_val_padded, y_val),
    epochs=10,
    batch_size=128,
    class_weight=class_weights,
    callbacks=[train_time_callback, model_checkpoint]
)

train_time = track_time(train_start)

# 2. Predictions
# Predictions on train set
train_pred_start = time.time()
train_predictions = model_BiLSTM.predict(X_train_padded)
train_pred_time = track_time(train_pred_start)

# Predictions on test set
test_pred_start = time.time()
test_predictions = model_BiLSTM.predict(X_test_padded)
test_pred_time = track_time(test_pred_start)

# Calculating accuracy scores
train_accuracy = calculate_accuracy(y_train, train_predictions)
test_accuracy = calculate_accuracy(y_test, test_predictions)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Training Time: {train_time:.2f} seconds")
print(f"Inference Time (training set): {train_pred_time:.2f} seconds")
print(f"Inference Time (testing set): {test_pred_time:.2f} seconds")

plot_training(history_BiLSTM)
eval_model(model_BiLSTM, X_train_padded, y_train, X_test_padded, y_test)

# With Attention
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D

# Class weight
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

# Initialize Learning Rate
learning_rate = 0.001 #0.01
optimizer = Adam(learning_rate=learning_rate)
#weight_decay = 0.001

# Define the input layer
inputs = keras.Input(shape=(max_length,), dtype="int32")

# Embedding layer
#x = layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs)
x = layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_length, trainable=True)(inputs)

# BiLSTM layer
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, return_state=False))(x)

# Attention mechanism

# Compute the importance of each step
e = layers.Dense(1, activation='tanh')(x)

# Flatten the importance scores
e = layers.Flatten()(e)

# Apply softmax to get the attention weights
a = layers.Activation('softmax', name='attention')(e)

# Multiply the attention weights with the LSTM output
temp = layers.RepeatVector(128)(a)  # Since it's BiLSTM, we've 2 * 64 = 128 dimensions
temp = layers.Permute([2, 1])(temp)
x = layers.Multiply()([x, temp])

# Post-attention LSTM (COULD BE REMOVE, TRY TO CHECK IF THE PERFORMANCE CHANGE)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, return_state=False))(x)

# Dense layer for classification
outputs = layers.Dense(1, activation='sigmoid')(x)

# Combine into a Model
model_BiLSTM_attention = keras.Model(inputs, outputs)

# Summary
model_BiLSTM_attention.summary()

# Compile the model
model_BiLSTM_attention.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
plot_model(model_BiLSTM_attention, show_shapes=True)

# Train the model
from sklearn.metrics import accuracy_score

# Train the model
epochs = 10  # Number of passes through the entire training dataset.
batch_size = 128   # 32, 64, 128, 256, 512 Number of samples used in each update of model weights.
train_time_mins = 60 # early time
train_time_callback = TrainForTime(train_time_mins=train_time_mins) # early time
#early_stopping = EarlyStopping(monitor='val_loss', patience=10) # Setting up Early Stopping
class_weights = class_weight_dict  # You should compute these weights based on your data distribution

dcnn_train_start = process_time()

history_BiLSTM_attention = model_BiLSTM_attention.fit(X_train_padded, y_train,
                    validation_data=(X_val_padded, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=class_weights,
                    callbacks=[train_time_callback, model_checkpoint])

dcnn_train_end = process_time()

# Obtain predicted labels from the model's predictions (train)
dcnn_inference_start = process_time()
train_predictions = model_BiLSTM_attention.predict(X_train_padded)
dcnn_inference_end = process_time()

# Convert predictions to class labels
train_labels_pred = (train_predictions > 0.5).astype("int32")

# Obtain predicted labels from the model's predictions (test)
dcnn_pred_start = process_time()
test_predictions = model_BiLSTM_attention.predict(X_test_padded)
dcnn_pred_end = process_time()

# Convert predictions to class labels
test_labels_pred = (test_predictions > 0.5).astype("int32")

# Time: Calculate training, inference, prediction time
dcnn_train_time = dcnn_train_end - dcnn_train_start
dcnn_inference_train_time = dcnn_inference_end - dcnn_inference_start
dcnn_inference_test_time = dcnn_pred_end - dcnn_pred_start

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, train_labels_pred)
test_accuracy = accuracy_score(y_test, test_labels_pred)

print("Train Accuracy: %f\nTest Accuracy: %f\nTraining Time: %f\nInference Time (training set): %f\nInference Time (testing set): %f" % \
      (train_accuracy, test_accuracy, dcnn_train_time, dcnn_inference_train_time, dcnn_inference_test_time))

# Evaluating the Model
plot_training(history_BiLSTM_attention)
eval_model(model_BiLSTM_attention, X_train_padded, y_train, X_test_padded, y_test, dcnn_train_time)

