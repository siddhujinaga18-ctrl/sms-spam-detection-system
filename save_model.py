



import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load Dataset
data = pd.read_csv(
    "dataset/spam.csv",
    encoding="latin-1"
)

# Clean Dataset
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert Labels
data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})

# Convert Text to Numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])

y = data['label']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# Save Model
with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save Vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model Saved Successfully!")