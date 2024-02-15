

import torch
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers

df= pd.read_csv('train.csv')
df=df.drop(['filename', 'length'], axis=1)

# map all labels to numbers
mapped_data = df.copy()
mapped_data.label = mapped_data.label.map({'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9})
mapped_data.head()
# Apply the mapping function to the class label column
#df['label'] = df['label'].apply(map_class_label)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['label'], axis=1), df['label'], test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode genre labels into numerical values
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

model = keras.Sequential([
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(80,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(80,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model on the test data
accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy on the test data: {accuracy[1] * 100:.2f}%')

test_df = pd.read_csv('test.csv')
resultdf = test_df.copy()
# Preprocess the test data
test_df = test_df.drop(['id', 'length'], axis=1)


# Normalize the test data
test_data = scaler.transform(test_df)

predictions = model.predict(test_data)

predictions

predictions = np.argmax(predictions,axis=1)
predictions

results=pd.DataFrame({'id':resultdf['id'], 'label':predictions})
results.to_csv('results.csv', index=False)

