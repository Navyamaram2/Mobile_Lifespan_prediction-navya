import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- “CSS” THEME CONSTANTS ---
BG_COLOR           = "#2E3440"  # window & frame background
FG_COLOR           = "#D8DEE9"  # labels, entry text
ENTRY_BG           = "#3B4252"  # entry field background
ENTRY_FG           = "#ECEFF4"  # entry field text
BUTTON_BG          = "#81A1C1"  # button background
BUTTON_FG          = BG_COLOR   # button text color
BUTTON_ACTIVE_BG   = "#5E81AC"  # button hover background
FONT_LABEL         = ("Helvetica", 12)
FONT_TITLE         = ("Helvetica", 14, "bold")
FONT_ENTRY         = ("Helvetica", 11)
FONT_BUTTON        = ("Arial", 12, "bold")

# --- LOAD MODEL & DATASET ---
with open("phone_lifespan_model.pkl", "rb") as file:
    model = pickle.load(file)

df = pd.read_csv(r"C:\Users\navya\OneDrive\Desktop\DM\Mobile_lifespan_app\Mobile_dataset.csv")
df.drop(columns=["User ID"], inplace=True)

categorical_cols = ["Device Model", "Operating System", "Gender"]
numeric_cols = [
    "App Usage Time (min/day)", "Screen On Time (hours/day)",
    "Battery Drain (mAh/day)", "Number of Apps Installed",
    "Data Usage (MB/day)", "Age", "User Behavior Class"
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

features = [
    "App Usage Time (min/day)", "Screen On Time (hours/day)",
    "Battery Drain (mAh/day)", "Number of Apps Installed",
    "Data Usage (MB/day)", "Age", "Gender", "User Behavior Class"
]

# --- PREDICTION FUNCTION ---
def predict():
    try:
        new_user = {
            "App Usage Time (min/day)": int(entry_app_usage.get()),
            "Screen On Time (hours/day)": float(entry_screen_time.get()),
            "Battery Drain (mAh/day)": int(entry_battery.get()),
            "Number of Apps Installed": int(entry_apps.get()),
            "Data Usage (MB/day)": int(entry_data.get()),
            "Age": int(entry_age.get()),
            "Gender": gender_var.get(),
            "User Behavior Class": int(entry_behavior.get()),
            "Device Model": entry_device.get(),
            "Operating System": entry_os.get()
        }
        # encode gender
        gender_map = {"Male": 1, "Female": 0}
        new_user["Gender"] = gender_map.get(new_user["Gender"], 0)

        new_df = pd.DataFrame([new_user])
        new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])

        pred = model.predict(new_df[features])[0]
        messagebox.showinfo("Prediction Result",
                            f"📱 Estimated Phone Lifespan: {pred:.2f} months")
    except Exception as e:
        messagebox.showerror("Input Error", str(e))


# --- BUILD UI ---
root = tk.Tk()
root.title("📱 Mobile Phone Lifespan Predictor")
root.configure(bg=BG_COLOR)
root.geometry("400x550")

# Title
tk.Label(root,
         text="Enter User Phone Usage Details",
         font=FONT_TITLE,
         bg=BG_COLOR,
         fg=FG_COLOR).pack(pady=15)

# Helper to create a labeled entry
def create_label_entry(parent, label_text):
    frame = tk.Frame(parent, bg=BG_COLOR)
    frame.pack(fill="x", padx=20, pady=5)
    tk.Label(frame,
             text=label_text,
             width=25,
             anchor="w",
             bg=BG_COLOR,
             fg=FG_COLOR,
             font=FONT_LABEL).pack(side="left")
    entry = tk.Entry(frame,
                     bg=ENTRY_BG,
                     fg=ENTRY_FG,
                     insertbackground=ENTRY_FG,
                     font=FONT_ENTRY,
                     bd=0,
                     highlightthickness=1,
                     highlightcolor=BUTTON_ACTIVE_BG)
    entry.pack(side="left", fill="x", expand=True)
    return entry

entry_app_usage   = create_label_entry(root, "App Usage Time (min/day):")
entry_screen_time = create_label_entry(root, "Screen On Time (hours/day):")
entry_battery     = create_label_entry(root, "Battery Drain (mAh/day):")
entry_apps        = create_label_entry(root, "Number of Apps Installed:")
entry_data        = create_label_entry(root, "Data Usage (MB/day):")
entry_age         = create_label_entry(root, "Age:")
entry_behavior    = create_label_entry(root, "User Behavior Class (1-5):")
entry_device      = create_label_entry(root, "Device Model:")
entry_os          = create_label_entry(root, "Operating System:")

# Gender dropdown
frm_gender = tk.Frame(root, bg=BG_COLOR)
frm_gender.pack(fill="x", padx=20, pady=5)
tk.Label(frm_gender,
         text="Gender:",
         width=25,
         anchor="w",
         bg=BG_COLOR,
         fg=FG_COLOR,
         font=FONT_LABEL).pack(side="left")
gender_var = tk.StringVar(value="Male")
opt = tk.OptionMenu(frm_gender, gender_var, "Male", "Female")
opt.config(bg=BUTTON_BG,
           fg=BUTTON_FG,
           activebackground=BUTTON_ACTIVE_BG,
           activeforeground=BUTTON_FG,
           font=FONT_ENTRY,
           bd=0,
           highlightthickness=0)
opt["menu"].config(bg=ENTRY_BG, fg=ENTRY_FG)  # style dropdown menu
opt.pack(side="left", fill="x", expand=True)

# Predict button
btn = tk.Button(root,
                text="Predict Lifespan",
                command=predict,
                bg=BUTTON_BG,
                fg=BUTTON_FG,
                activebackground=BUTTON_ACTIVE_BG,
                activeforeground=BUTTON_FG,
                font=FONT_BUTTON,
                bd=0,
                highlightthickness=0,
                padx=10,
                pady=5)
btn.pack(pady=25)

root.mainloop()
