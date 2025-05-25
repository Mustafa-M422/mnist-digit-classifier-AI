# digit_classifier.py
# COMP 472 – Mini Project 1
# Author(s): Mostafa Mohamed (400XXXXXX), [Partner Name] (400YYYYYY)
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ========== 1. Load the Dataset ==========
digits = load_digits()
X, y = digits.data, digits.target

print("Dataset shape:", X.shape)
print("Target labels:", np.unique(y))

# ========== 2. Visualize Sample Digits ==========
plt.figure(figsize=(6, 4))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f'Label: {y[i]}')
    plt.axis('off')
plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/sample_digits.png")  # Save for GitHub README
plt.show()

# ========== 3. Normalize the Data ==========
X = X / 16.0  # since pixel values range from 0 to 16

# ========== 4. Split into Training and Testing ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== 5. Train the Logistic Regression Model ==========
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# ========== 6. Make Predictions ==========
y_pred = model.predict(X_test)

# ========== 7. Evaluate the Model ==========
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nClassification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Save reports to text files (optional)
with open("results/classification_report.txt", "w") as f:
    f.write(report)

with open("results/confusion_matrix.txt", "w") as f:
    f.write(str(conf_matrix))
