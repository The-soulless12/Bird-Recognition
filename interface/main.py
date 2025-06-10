import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from playsound import playsound
import glob

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BIRD RECOGNITION")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        # On appelle le modèle pré-entraîné
        self.model = load_model(os.path.join('../model', 'E-30-35-98,83+.keras'))
        images_dir = '../images'
        self.class_names = sorted([
            d for d in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, d))
        ])
        self.selected_image_path = None

        self.bg_image = Image.open("fond.png").resize((600, 400))
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        self.bg_label = tk.Label(root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)        

        self.image_label = tk.Label(root, bg="#fce7eb")
        self.image_label.place(x=50, y=50)

        self.select_button = tk.Button(root, text="SELECT IMAGE", font=("Comic Sans MS", 10), command=self.select_image)
        self.select_button.place(x=60, y=270)

        self.predict_button = tk.Button(root, text="PREDICT", font=("Comic Sans MS", 10), command=self.predict_image)
        self.predict_button.place(x=180, y=270)

        self.result_label = tk.Label(root, text="", font=("Comic Sans MS", 12), bg="#eec2c9", width=30, wraplength=200,  justify="center", anchor="center" )
        self.result_label.place(x=50, y=320, width=200)

        self.sound_button = tk.Button(root, text="SON", font=("Comic Sans MS", 10), command=self.play_sound)
        self.sound_button.place(x=550, y=360)
        self.sound_button.config(state="disabled") 

    def create_rounded_box(self, w, h, r, color_hex):
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0, 0, w, h), radius=r, fill=color_hex)
        return img

    def apply_rounded_mask(self, img, r):
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0) + img.size, radius=r, fill=255)
        img.putalpha(mask)
        return img

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.selected_image_path = file_path
            img = Image.open(file_path).convert("RGBA")
            img = ImageOps.fit(img, (200, 200), method=Image.Resampling.LANCZOS)
            img = self.apply_rounded_mask(img, 5)
            self.tk_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_image)
            self.result_label.config(text="")

    def prepare_image(self, img_path, target_size=(224, 224)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_image(self):
        if self.selected_image_path:
            img_prepared = self.prepare_image(self.selected_image_path)
            preds = self.model.predict(img_prepared, verbose=0)
            predicted_index = np.argmax(preds)

            if 0 <= predicted_index < len(self.class_names):
                class_name = self.class_names[predicted_index]
                self.predicted_class_name = class_name  
                self.result_label.config(text=f"{class_name}")
                self.sound_button.config(state="normal")  
            else:
                self.result_label.config(text="Classe inconnue.")
                self.sound_button.config(state="disabled")
        else:
            self.result_label.config(text="Veuillez sélectionner une image.")

    def play_sound(self):
        if hasattr(self, "predicted_class_name"):
            base_path = os.path.join("..", "sons")
            pattern = os.path.join(base_path, f"{self.predicted_class_name}.*")
            matches = glob.glob(pattern)

            if matches:
                try:
                    playsound(matches[0]) 
                except Exception as e:
                    print(f"Erreur lors de la lecture du son : {e}")
            else:
                print(f"Aucun fichier audio trouvé pour : {self.predicted_class_name}")

root = tk.Tk()
app = ImageApp(root)
root.mainloop()