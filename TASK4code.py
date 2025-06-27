#Used Google collab notebook

#first cell:

from google.colab import files
uploaded = files.upload()           #uploaded the SMSSpamCollection.csv

#second cell:
import pandas as pd

# Load tab-separated file with no header
df = pd.read_csv("SMSSpamCollection.csv", sep='\t', header=None, names=['label', 'message'])

# Drop missing values
df.dropna(subset=['label', 'message'], inplace=True)

# Map labels
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Prepare features and labels
X = df['message'].fillna('')
y = df['label_num'].astype(int)

# Check rows
print(f"✅ Loaded {len(df)} messages")


#third cell:

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#fouth cell:
# Split the data-
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#fifth cell:
# Vectorize-
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#sixth cell:
# Train model-
model = MultinomialNB()
model.fit(X_train_vec, y_train)

#seventh cell:
# Predict and evaluate-
y_pred = model.predict(X_test_vec)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


#eighth cell:
# Plot
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()






