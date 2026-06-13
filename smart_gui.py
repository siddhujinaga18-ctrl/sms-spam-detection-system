import tkinter as tk
import pickle

# Load Saved Model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load Saved Vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Prediction Function
def predict_spam():
    message = text_box.get("1.0", tk.END)

    test_vector = vectorizer.transform([message])

    prediction = model.predict(test_vector)

    if prediction[0] == 1:
        result_label.config(
            text="🚨 SPAM MESSAGE"
        )
    else:
        result_label.config(
            text="✅ HAM MESSAGE"
        )

# Main Window
window = tk.Tk()
window.title("SMS Spam Detection System")
window.geometry("700x500")
window.configure(bg="#EAF6F6")

# Title
title = tk.Label(
    window,
    text="SMS Spam Detection System",
    font=("Arial", 22, "bold"),
    bg="#EAF6F6"
)
title.pack(pady=20)

# Instruction Label
instruction = tk.Label(
    window,
    text="Enter SMS Message Below",
    font=("Arial", 12),
    bg="#EAF6F6"
)
instruction.pack()

# Text Box
text_box = tk.Text(
    window,
    height=10,
    width=60,
    font=("Arial", 12)
)
text_box.pack(pady=15)

# Predict Button
button = tk.Button(
    window,
    text="Predict",
    font=("Arial", 14, "bold"),
    width=15,
    height=2,
    command=predict_spam
)
button.pack(pady=10)

# Result Label
result_label = tk.Label(
    window,
    text="Waiting for Prediction...",
    font=("Arial", 16, "bold"),
    bg="#EAF6F6"
)
result_label.pack(pady=20)

# Run Window
window.mainloop()