import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk

# Load Dataset
data = pd.read_csv(
    "dataset/spam.csv",
    encoding="latin-1"
)

# Clean Dataset
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

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

accuracy = model.score(X_test, y_test)

# Prediction Function
def predict_spam():
    message = text_box.get("1.0", tk.END)

    test_vector = vectorizer.transform([message])

    prediction = model.predict(test_vector)

    if prediction[0] == 1:
        result_label.config(text="🚨 SPAM MESSAGE")
    else:
        result_label.config(text="✅ HAM MESSAGE")


# GUI Window
window = tk.Tk()
window.title("SMS Spam Detection")
window.geometry("500x400")

title_label = tk.Label(
    window,
    text="SMS Spam Detection",
    font=("Arial", 18, "bold")
)
title_label.pack(pady=10)

accuracy_label = tk.Label(
    window,
    text=f"Model Accuracy: {round(accuracy * 100, 2)}%"
)
accuracy_label.pack()

text_box = tk.Text(
    window,
    height=8,
    width=50
)
text_box.pack(pady=10)

predict_button = tk.Button(
    window,
    text="Predict",
    command=predict_spam
)
predict_button.pack()

result_label = tk.Label(
    window,
    text="",
    font=("Arial", 14, "bold")
)
result_label.pack(pady=20)

window.mainloop()