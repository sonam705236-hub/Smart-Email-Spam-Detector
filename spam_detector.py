import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- File Handling ---
CSV_FILE = 'emails_data.csv'

def load_data():
    # Agar file nahi hai, toh basic data ke saath banao
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        initial_df = pd.DataFrame({
            'text': ['click here for free prize', 'hello how are you', 'win cash now', 'meeting at five'],
            'label': ['spam', 'ham', 'spam', 'ham']
        })
        # latin-1 encoding sabse safe hai symbols ke liye
        initial_df.to_csv(CSV_FILE, index=False, encoding='latin-1')
        return initial_df
    else:
        try:
            # Pehle utf-8 try karo
            return pd.read_csv(CSV_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            # Agar error aaye (jo ki pound symbol ki wajah se aa raha hai), toh latin-1 use karo
            return pd.read_csv(CSV_FILE, encoding='latin-1')

df = load_data()
cv = CountVectorizer()
model = MultinomialNB()

def train_model():
    global X, y, df
    if df.empty: return
    df = df.dropna()
    X = cv.fit_transform(df['text'])
    y = df['label']
    model.fit(X, y)

train_model()

# --- Functions ---
def auto_refresh(event=None):
    user_text = email_entry.get("1.0", tk.END).strip()
    if len(user_text) < 2: 
        result_label.config(text="Waiting for input...", fg="black")
        return
    
    try:
        vec = cv.transform([user_text])
        res = model.predict(vec)[0]
        color = "#ff4757" if res == 'spam' else "#2ed573"
        result_label.config(text=f"PREDICTION: {res.upper()}", fg=color)
    except:
        pass

def teach_and_save(label):
    global df
    user_text = email_entry.get("1.0", tk.END).strip()
    
    if len(user_text) < 3:
        messagebox.showwarning("Error", "Pehle kuch message likho!")
        return

    new_data = pd.DataFrame([{'text': user_text, 'label': label}])
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Save hamesha latin-1 mein karo symbols ke liye
    df.to_csv(CSV_FILE, index=False, encoding='latin-1')
    
    train_model()
    email_entry.delete("1.0", tk.END)
    result_label.config(text="Model Trained Permanently!", fg="#2f3542")
    messagebox.showinfo("Success", f"Saved to CSV as {label.upper()}!")

# --- GUI Design ---
root = tk.Tk()
root.title("AI Smart Spam Detector")
root.geometry("600x500")
root.config(bg="#f1f2f6")

tk.Label(root, text="🛡️ Smart AI Email Classifier", font=("Arial", 22, "bold"), bg="#f1f2f6", fg="#2f3542").pack(pady=25)
email_entry = tk.Text(root, height=8, width=55, font=("Arial", 11), padx=10, pady=10)
email_entry.pack(pady=10)
email_entry.bind("<KeyRelease>", auto_refresh)

result_label = tk.Label(root, text="Start typing...", font=("Arial", 16, "bold"), bg="#f1f2f6")
result_label.pack(pady=20)

btn_frame = tk.Frame(root, bg="#f1f2f6")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="🚨 MARK AS SPAM", command=lambda: teach_and_save('spam'), 
          bg="#ff4757", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10).pack(side=tk.LEFT, padx=15)

tk.Button(btn_frame, text="✅ MARK AS SAFE", command=lambda: teach_and_save('ham'), 
          bg="#2ed573", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10).pack(side=tk.LEFT, padx=15)

root.mainloop()
