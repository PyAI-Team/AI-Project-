# Breast Cancer Detection Using AI

This project uses a neural network with TensorFlow to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous).

## 📊 Dataset Preparation
- We load the dataset using Pandas and separate features (`X`) and labels (`Y`).
- `X`: Contains features like radius, texture, and area.
- `Y`: Indicates whether the tumor is benign (0) or malignant (1).

## 🧠 Building the AI Model
- **Splitting Data**: We split the data into training (80%) and testing (20%) sets using Scikit-Learn.
- **Neural Network**: We build a model using Keras with two hidden layers (256 neurons each).

## ✅ Training & Testing the machine
- The model is trained using 1,000 epochs to identify patterns.
- We evaluate the model using the testing data to check its accuracy.

## 📝 Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load dataset
dataset = pd.read_csv('cancer.csv')

# Split dataset into features and target
x = dataset.drop(columns=['diagnosis(1=m, 0=b)'])
y = dataset['diagnosis(1=m, 0=b)']

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=1000)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')
```
## 🖼 Output

# Below is a screenshot showing the key results of the model training and testing process, including the accuracy :
![Output](https://github.com/user-attachments/assets/40c48f5b-d3f8-4041-8ccc-d4282d37e08e)







