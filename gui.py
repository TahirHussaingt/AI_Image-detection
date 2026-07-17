import tkinter as tk
from tkinter import filedialog, Label, Frame
from PIL import Image, ImageTk
import pillow_heif
import tensorflow as tf
import numpy as np
import os

pillow_heif.register_heif_opener()

model = tf.keras.models.load_model("image_detector_model.keras")

img_height = 224
img_width = 224
class_names = ['FAKE', 'REAL']

history = []  # keep track of recent predictions

def predict_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction > 0.5:
        return class_names[1], prediction * 100
    else:
        return class_names[0], (1 - prediction) * 100

def get_confidence_label(confidence):
    if confidence >= 85:
        return "High confidence", "#2e7d32"      # green
    elif confidence >= 65:
        return "Moderate confidence", "#f9a825"  # amber
    else:
        return "Low confidence - verify manually", "#c62828"  # red

def update_history(filename, label, confidence):
    history.insert(0, f"{filename} → {label} ({confidence:.1f}%)")
    if len(history) > 5:
        history.pop()
    history_text.set("\n".join(history))

def process_image(file_path):
    if not os.path.exists(file_path):
        return
    if os.path.getsize(file_path) == 0:
        result_label.config(text="⚠️ This file is empty (0 bytes).\nTry a different image.", fg="red")
        image_label.config(image="")
        return

    try:
        img = Image.open(file_path)
        img = img.convert("RGB")
        img_display = img.resize((260, 260))
        img_tk = ImageTk.PhotoImage(img_display)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        label, confidence = predict_image(file_path)
        conf_text, conf_color = get_confidence_label(confidence)

        color = "#2e7d32" if label == "REAL" else "#c62828"
        result_label.config(text=f"Prediction: {label}", fg=color)
        confidence_label.config(text=f"Confidence: {confidence:.2f}%  ({conf_text})", fg=conf_color)

        # Update confidence bar
        bar_width = int((confidence / 100) * 260)
        confidence_canvas.delete("bar")
        confidence_canvas.create_rectangle(0, 0, bar_width, 20, fill=conf_color, outline="", tags="bar")

        update_history(os.path.basename(file_path), label, confidence)

    except Exception:
        result_label.config(text="⚠️ Couldn't read this image.\nTry a different file.", fg="red")
        confidence_label.config(text="")
        image_label.config(image="")

def browse_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.heic *.webp *.bmp")]
    )
    if file_path:
        process_image(file_path)

# ---- Drag and drop support ----
def on_drop(event):
    file_path = event.data.strip("{}")  # handles paths with spaces
    process_image(file_path)

# ---- Build the window (tkinterdnd2 needs its own Tk class for drag-and-drop) ----
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    root = TkinterDnD.Tk()
    dnd_available = True
except ImportError:
    root = tk.Tk()
    dnd_available = False

root.title("AI Image Detector")
root.geometry("420x680")
root.configure(bg="#f5f5f5")

title_label = Label(root, text="🔍 AI Image Detector", font=("Segoe UI", 18, "bold"), bg="#f5f5f5", fg="#212121")
title_label.pack(pady=(20, 5))

subtitle_label = Label(root, text="Detects real photos vs photorealistic AI images", font=("Segoe UI", 10), bg="#f5f5f5", fg="#616161")
subtitle_label.pack(pady=(0, 15))

browse_btn = tk.Button(root, text="📁 Choose Image", command=browse_and_predict,
                        font=("Segoe UI", 12), bg="#1976d2", fg="white",
                        activebackground="#1565c0", activeforeground="white",
                        relief="flat", padx=20, pady=8, cursor="hand2")
browse_btn.pack(pady=10)

drop_hint = Label(root, text="(or drag & drop an image below)", font=("Segoe UI", 9, "italic"), bg="#f5f5f5", fg="#9e9e9e")
drop_hint.pack()

# Image display frame (also acts as drop zone)
image_frame = Frame(root, width=260, height=260, bg="white", relief="groove", bd=2)
image_frame.pack(pady=15)
image_frame.pack_propagate(False)

image_label = Label(image_frame, bg="white")
image_label.pack(expand=True)

result_label = Label(root, text="No image selected yet", font=("Segoe UI", 15, "bold"), bg="#f5f5f5")
result_label.pack(pady=(10, 2))

confidence_label = Label(root, text="", font=("Segoe UI", 11), bg="#f5f5f5")
confidence_label.pack(pady=(0, 5))

confidence_canvas = tk.Canvas(root, width=260, height=20, bg="#e0e0e0", highlightthickness=0)
confidence_canvas.pack(pady=(0, 20))

# History section
history_title = Label(root, text="Recent Predictions", font=("Segoe UI", 11, "bold"), bg="#f5f5f5", fg="#424242")
history_title.pack()

history_text = tk.StringVar()
history_label = Label(root, textvariable=history_text, font=("Segoe UI", 9), bg="#f5f5f5", fg="#616161", justify="left")
history_label.pack(pady=(5, 20))

# ---- Enable drag-and-drop ----
if dnd_available:
    image_frame.drop_target_register(DND_FILES)
    image_frame.dnd_bind('<<Drop>>', on_drop)
else:
    drop_hint.config(text="(drag & drop unavailable - install tkinterdnd2)")

root.mainloop()