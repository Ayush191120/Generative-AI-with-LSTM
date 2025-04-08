#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import string
import requests


# # Loading Data

# In[2]:


url = "https://www.gutenberg.org/ebooks/4255.txt.utf-8"
response = requests.get(url)
data = response.text
print(f"Dataset length: {len(data)} characters")
print(data[:500])


# # Data Preprocessing

# In[3]:


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    return text

cleaned_text = preprocess_text(data)
print(cleaned_text[:500])


# In[4]:


# Tokenize the text (character-level)
chars = sorted(list(set(cleaned_text)))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

vocab_size = len(chars)
print("Total Characters:", len(cleaned_text))
print("Unique Characters:", vocab_size)
print("Character to integer mapping", char_to_int)


# In[5]:


# Prepare sequences of characters as input and next character as output
seq_length = 100 # Length of input sequences
step = 3 # Step to move while creating sequences

sequences = []
next_chars = []

for i in range(0, len(cleaned_text) - seq_length, step):
    sequences.append(cleaned_text[i:i + seq_length])
    next_chars.append(cleaned_text[i + seq_length])
    
    print(f"Number Of sequences: {len(sequences)}")
    print("Example sequence:", sequences[0])
    print("Next character:", next_chars[0])


# In[6]:


# Convert sequences to numerical representation
X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool_)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool_)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1

print("X shape:", X.shape)
print("y shape:", y.shape)


# # Model Design

# In[7]:


model = Sequential([
    LSTM(256, input_shape=(seq_length, vocab_size), return_sequences=True),
    LSTM(256),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


# # Model Training

# In[8]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define callbacks
checkpoint = ModelCheckpoint("text_gen_model.h5", monitor='loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

history = model.fit(X, y, batch_size=128, epochs=20, callbacks=[checkpoint, early_stopping])


# In[9]:


# Plot training loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


# # Text Generation

# In[10]:


def generate_text(seed, num_chars=200, temperature=0.5):
    generated = seed
    for i in range(num_chars):
        # Prepare the input sequence
        x_pred = np.zeros((1, seq_length, vocab_size))
        for t, char in enumerate(seed):
            x_pred[0, t, char_to_int[char]] = 1.
        
         # Make prediction
        preds = model.predict(x_pred, verbose=0)[0]
        
        # Apply temperature
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        # Sample next character
        next_index = np.random.choice(range(vocab_size), p=preds)
        next_char = int_to_char[next_index]
        
        # Update seed and generated text
        seed = seed[1:] + next_char
        generated += next_char
        
    return generated


# In[11]:


# Generate some text
seed_sequence = cleaned_text[:seq_length]  # Get first seq_length characters as seed
print("Seed sequence:", seed_sequence)

generated_text = generate_text(seed_sequence, num_chars=500, temperature=0.5)
print("\nGenerated text:")
print(generated_text)


# In[12]:


# Try different temperatures
print("Temperature 0.2 (more conservative):")
print(generate_text(seed_sequence, num_chars=500, temperature=0.2))

print("\nTemperature 1.0 (more diverse):")
print(generate_text(seed_sequence, num_chars=500, temperature=1.0))


# In[ ]:




