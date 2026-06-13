import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv(
    "dataset/spam.csv",
    encoding="latin-1"
)

# Keep only required columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['label', 'message']

# Convert labels to numbers
data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])

# Target column
y = data['label']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train model
model = MultinomialNB()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Check accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# User input
message = "had your dinner?"

# Convert user message into numbers
test_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(test_vector)

# Show result
if prediction[0] == 1:
    print("\nResult: SPAM")
else:
    print("\nResult: HAM")

