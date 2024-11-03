import numpy as np
import pandas as pd
import tensorflow as tf
import keras

# Load your data
data = pd.read_excel('summarizer/Data/news.xlsx') 

texts = data['text'].values  
summaries = data['summary'].values  

# Define parameters
max_tokens = 10_000 
max_len_text = 500 
max_len_summary = 100  

# Create TextVectorization layers
text_vectorizer = keras.layers.TextVectorization(max_tokens=max_tokens, output_mode='int', output_sequence_length=max_len_text)
summary_vectorizer = keras.layers.TextVectorization(max_tokens=max_tokens, output_mode='int', output_sequence_length=max_len_summary)

# Adapt the vectorizers to the data
text_vectorizer.adapt(texts)
summary_vectorizer.adapt(summaries)

# Vectorize the texts and summaries
X = text_vectorizer(texts)
y = summary_vectorizer(summaries)

# Encoder
encoder_inputs = keras.layers.Input(shape=(max_len_text,))
encoder_embedding = keras.layers.Embedding(input_dim=max_tokens, output_dim=128)(encoder_inputs)
encoder_lstm = keras.layers.LSTM(128, return_sequences=True)(encoder_embedding)

# Decoder
decoder_inputs = keras.layers.Input(shape=(max_len_summary,))
decoder_embedding = keras.layers.Embedding(input_dim=max_tokens, output_dim=128)(decoder_inputs)
decoder_lstm = keras.layers.LSTM(128, return_sequences=True)(decoder_embedding)

# Attention Layer
attention_layer = keras.layers.Attention()([encoder_lstm, decoder_lstm])
decoder_concat_input = keras.layers.Concatenate(axis=-1)([decoder_lstm, attention_layer])

# Output Layer
decoder_dense = keras.layers.Dense(max_tokens, activation='softmax')(decoder_concat_input)

# Model Definition
model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_dense)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare decoder input data (shifted summaries)
y_shifted = np.zeros_like(y)
y_shifted[:, :-1] = y[:, 1:]

model.fit([X, y_shifted], np.expand_dims(y, -1), batch_size=64, epochs=125)
model.save('summariser.keras')