from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import pandas as pd
import tensorflow as tf
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Data Preprocessing
df = pd.read_json(r"News_Category_Dataset_v3.json", lines=True)
df = df[['headline', 'category']]

# Step 2: Text Vectorization
max_words = 2000
max_len = 50
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['headline'])
sequences = tokenizer.texts_to_sequences(df['headline'])
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Step 3: Label Encoding
labelencoder = LabelEncoder()
df['category_encoded'] = labelencoder.fit_transform(df['category'])

# Step 4: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['category_encoded'], test_size=0.2, random_state=42)

# Step 5: Model Building with LSTM
model = Sequential([
    Embedding(max_words, 50, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(len(df['category'].unique()), activation='softmax')
])

# Step 6: Model Compilation
model.compile(loss=SparseCategoricalCrossentropy(),
              optimizer=Adam(),
              metrics=['accuracy'])

# Step 7: Model Training
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)
print(history.history)
# Specify the file path (change this to your preferred directory)
model_file_path = r"News_Category_Model.h5"

# Save the model
model.save(r"News_Category_Model.h5")