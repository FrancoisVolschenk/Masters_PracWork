import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import infer

MODEL_DIR = "./Fingerprint_Synthesis/model"


class ModelUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Inference UI")

        # State
        self.selected_model = None
        self.loaded_model = None
        self.input_image = None
        self.output_image = None

        # Frame: Model selection
        self.model_frame = tk.Frame(root, padx=10, pady=10)
        self.model_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(self.model_frame, text="Select a model:").pack(anchor="w")

        self.model_listbox = tk.Listbox(
            self.model_frame, height=5, exportselection=False
        )
        self.model_listbox.pack(fill=tk.X)

        self.load_models()

        self.load_model_btn = tk.Button(
            self.model_frame, text="Load Model", command=self.load_model
        )
        self.load_model_btn.pack(pady=5)

        # Frame: Actions (hidden until model loaded)
        self.action_frame = tk.Frame(root, padx=10, pady=10)
        self.select_img_btn = tk.Button(
            self.action_frame, text="Select Input Image", command=self.select_image
        )

        # Frame: Display
        self.display_frame = tk.Frame(root, padx=10, pady=10)
        self.display_frame.pack(fill=tk.BOTH, expand=True)

        # Two labels side by side
        self.original_label = tk.Label(self.display_frame)
        self.original_label.pack(side=tk.LEFT, padx=5)

        self.reconstructed_label = tk.Label(self.display_frame)
        self.reconstructed_label.pack(side=tk.LEFT, padx=5)

    def load_models(self):
        """Populate the listbox with models in MODEL_DIR."""
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        models = os.listdir(MODEL_DIR)
        self.model_listbox.delete(0, tk.END)
        for m in models:
            self.model_listbox.insert(tk.END, m)

    def load_model(self):
        """Load the selected model (stub)."""
        try:
            idx = self.model_listbox.curselection()[0]
            self.selected_model = self.model_listbox.get(idx)
            # TODO: Hook into your real model-loading logic
            self.loaded_model = infer.load_model(self.selected_model)
            print(f"Loading model: {self.selected_model}")
            messagebox.showinfo("Model Loaded", f"Loaded model: {self.selected_model}")

            # Show action buttons
            self.action_frame.pack(side=tk.TOP, fill=tk.X)
            self.select_img_btn.pack()
        except IndexError:
            messagebox.showwarning("No Selection", "Please select a model first.")

    def select_image(self):
        """Open a file dialog to select an input image."""
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp")]
        path = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        if path:
            self.input_image = path
            # self.show_image(img_path=path)

            # Run inference (stub)
            self.run_inference(path)

    def run_inference(self, img_path):
        """Run inference using the loaded model (stub)."""
        print(f"Running inference on: {img_path} with {self.selected_model}")
        original_img, reconstructed_img = infer.process_image(
            self.loaded_model, img_path
        )
        # TODO: Replace with actual inference, save result to self.output_image
        # For now, just re-display the input
        self.show_images(original_img, reconstructed_img)

    def show_images(self, original, reconstructed):
        # Resize both to fit UI
        original = original.resize((400, 400))
        reconstructed = reconstructed.resize((400, 400))

        # Convert to Tk images (must keep references!)
        self.tk_original = ImageTk.PhotoImage(original)
        self.tk_reconstructed = ImageTk.PhotoImage(reconstructed)

        # Update the labels
        self.original_label.config(image=self.tk_original)
        self.reconstructed_label.config(image=self.tk_reconstructed)


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelUI(root)
    root.mainloop()
