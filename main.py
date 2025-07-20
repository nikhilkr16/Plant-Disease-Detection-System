import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Make sure your final utils.py file is in the same directory
from utils import EnsemblePredictor 

class PlantDiseaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Detection System (Ensemble)")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize the EnsemblePredictor with the paths to your trained models
        try:
            model_paths = [
                'models/efficientnetv2_best.h5',
                'models/resnet50v2_best.h5'
            ]
            self.predictor = EnsemblePredictor(model_paths)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load models: {e}\nPlease make sure the trained .h5 files are in the 'models' folder.")
            self.root.destroy()
            return
        
        self.current_pil_image = None
        self.create_widgets()
        
    def create_widgets(self):
        """Create and layout all the GUI widgets"""
        title_label = tk.Label(self.root, text="🌱 Plant Disease Detection System 🌱", font=("Arial", 24, "bold"), bg='#f0f0f0', fg='#2e7d32')
        title_label.pack(pady=20)
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        image_frame = ttk.LabelFrame(left_frame, text="Image", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.image_label = tk.Label(image_frame, text="No image selected", bg='white', width=40, height=20, relief=tk.SUNKEN)
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        upload_btn = ttk.Button(button_frame, text="📁 Upload Image", command=self.upload_image)
        upload_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.predict_btn = ttk.Button(button_frame, text="🔍 Predict Disease", command=self.predict_disease, state=tk.DISABLED)
        self.predict_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = ttk.Button(button_frame, text="🗑️ Clear", command=self.clear_results)
        clear_btn.pack(side=tk.LEFT)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        results_frame = ttk.LabelFrame(right_frame, text="Prediction Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_frame, width=50, height=23, wrap=tk.WORD, font=("Arial", 11), bg='#ffffff', relief=tk.SUNKEN, borderwidth=1)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        chart_frame = ttk.LabelFrame(right_frame, text="Confidence Chart", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.chart_canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Please upload an image.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_image(self):
        """Handle image upload and display"""
        file_path = filedialog.askopenfilename(title="Select Plant Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                self.current_pil_image = Image.open(file_path)
                display_image = self.current_pil_image.copy()
                display_image.thumbnail((500, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(display_image)
                
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                
                self.predict_btn.configure(state=tk.NORMAL)
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                self.clear_results(keep_image=True)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
                self.status_var.set("Error loading image")
    
    def predict_disease(self):
        """Run prediction on the uploaded image"""
        if not self.current_pil_image:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        
        try:
            self.status_var.set("Predicting with ensemble model...")
            self.root.update()
            
            result = self.predictor.predict(pil_image=self.current_pil_image)
            
            self.display_results(result)
            self.plot_confidence_chart(result)
            self.status_var.set("Prediction completed successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {str(e)}")
            self.status_var.set("Prediction error")
    
    def display_results(self, result):
        """Format and show the prediction results in the text area"""
        self.results_text.delete(1.0, tk.END)
        
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        
        self.results_text.insert(tk.END, "🔍 PREDICTION RESULTS\n" + "="*50 + "\n\n")
        self.results_text.insert(tk.END, f"🏆 Predicted Disease: {predicted_class.replace('_', ' ')}\n")
        self.results_text.insert(tk.END, f"📊 Confidence: {confidence:.2%}\n\n")
        
        self.results_text.insert(tk.END, "📈 Top Predictions:\n" + "-"*30 + "\n")
        all_preds = result['all_predictions']
        sorted_preds = sorted(all_preds.items(), key=lambda item: item[1], reverse=True)
        
        for i, (class_name, conf) in enumerate(sorted_preds[:3], 1):
            self.results_text.insert(tk.END, f"{i}. {class_name.replace('_', ' ')}: {conf:.2%}\n")
        
        disease_info = self.predictor.get_disease_info(predicted_class)
        self.results_text.insert(tk.END, "\n🔬 Disease Information:\n" + "-"*30 + "\n")
        self.results_text.insert(tk.END, f"Description: {disease_info.get('description', 'N/A')}\n\n")
        self.results_text.insert(tk.END, f"Symptoms: {disease_info.get('symptoms', 'N/A')}\n\n")
        self.results_text.insert(tk.END, f"Treatment: {disease_info.get('treatment', 'N/A')}\n")

    def plot_confidence_chart(self, result):
        """Generate and display the confidence bar chart"""
        self.ax.clear()
        
        all_preds = result['all_predictions']
        sorted_preds = sorted(all_preds.items(), key=lambda x: x[1], reverse=True)[:5]
        
        classes = [item[0].replace('_', ' ') for item in sorted_preds]
        confidences = [item[1] * 100 for item in sorted_preds]
        
        bars = self.ax.barh(classes, confidences, color='#4CAF50')
        self.ax.set_xlabel('Confidence (%)')
        self.ax.set_title('Ensemble Prediction Confidence')
        self.ax.set_xlim(0, 100)
        self.ax.invert_yaxis()
        
        for bar in bars:
            width = bar.get_width()
            self.ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        self.chart_canvas.draw()
    
    def clear_results(self, keep_image=False):
        """Reset the results and optionally the image"""
        if not keep_image:
            self.current_pil_image = None
            self.image_label.configure(image="", text="No image selected")
            self.image_label.image = None
            self.predict_btn.configure(state=tk.DISABLED)
            self.status_var.set("Ready. Please upload an image.")

        self.results_text.delete(1.0, tk.END)
        self.ax.clear()
        self.ax.set_title('Prediction Confidence')
        self.chart_canvas.draw()

def main():
    root = tk.Tk()
    app = PlantDiseaseGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()