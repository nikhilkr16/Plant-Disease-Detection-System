import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from utils import PlantDiseasePredictor
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PlantDiseaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize predictor
        self.predictor = PlantDiseasePredictor()
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create and layout GUI widgets"""
        # Main title
        title_label = tk.Label(
            self.root, 
            text="🌱 Plant Disease Detection System 🌱", 
            font=("Arial", 24, "bold"),
            bg='#f0f0f0',
            fg='#2e7d32'
        )
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left frame for image and buttons
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Image display frame
        image_frame = ttk.LabelFrame(left_frame, text="Image", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Image label
        self.image_label = tk.Label(
            image_frame, 
            text="No image selected", 
            bg='white',
            width=40, 
            height=20,
            relief=tk.SUNKEN
        )
        self.image_label.pack(expand=True)
        
        # Buttons frame
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Upload button
        upload_btn = ttk.Button(
            button_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            style="Accent.TButton"
        )
        upload_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Predict button
        self.predict_btn = ttk.Button(
            button_frame,
            text="🔍 Predict Disease",
            command=self.predict_disease,
            state=tk.DISABLED
        )
        self.predict_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear button
        clear_btn = ttk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_results
        )
        clear_btn.pack(side=tk.LEFT)
        
        # Right frame for results
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        # Results frame
        results_frame = ttk.LabelFrame(right_frame, text="Prediction Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results text
        self.results_text = tk.Text(
            results_frame,
            width=50,
            height=23,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg='#ffffff',
            relief=tk.SUNKEN,
            borderwidth=1
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Confidence chart frame
        chart_frame = ttk.LabelFrame(right_frame, text="Confidence Chart", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Chart canvas
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.chart_canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_image(self):
        """Upload and display image"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Plant Image",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # Load and display image
                self.current_image_path = file_path
                image = Image.open(file_path)
                
                # Resize image for display
                display_size = (400, 300)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image)
                
                # Update image label
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # Keep a reference
                
                # Enable predict button
                self.predict_btn.configure(state=tk.NORMAL)
                
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                
                # Clear previous results
                self.results_text.delete(1.0, tk.END)
                self.ax.clear()
                self.chart_canvas.draw()
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
                self.status_var.set("Error loading image")
    
    def predict_disease(self):
        """Predict disease from uploaded image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        
        # Remove the check for self.predictor.model is None
        # Always allow prediction (mock or real)
        try:
            self.status_var.set("Predicting...")
            self.root.update()
            
            # Make prediction
            result, message = self.predictor.predict_disease(image_path=self.current_image_path)
            
            if result is None:
                messagebox.showerror("Error", f"Prediction failed: {message}")
                self.status_var.set("Prediction failed")
                return
            
            # Display results
            self.display_results(result)
            self.plot_confidence_chart(result)
            
            self.status_var.set("Prediction completed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {str(e)}")
            self.status_var.set("Prediction error")
    
    def display_results(self, result):
        """Display prediction results in text widget"""
        self.results_text.delete(1.0, tk.END)
        
        # Main prediction
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        
        self.results_text.insert(tk.END, "🔍 PREDICTION RESULTS\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")
        
        self.results_text.insert(tk.END, f"🏆 Predicted Disease: {predicted_class}\n")
        self.results_text.insert(tk.END, f"📊 Confidence: {confidence:.2%}\n\n")
        
        # Top 3 predictions
        self.results_text.insert(tk.END, "📈 Top 3 Predictions:\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        
        for i, (class_name, conf) in enumerate(result['top_3_predictions'], 1):
            self.results_text.insert(tk.END, f"{i}. {class_name}: {conf:.2%}\n")
        
        # Disease information
        disease_info = self.predictor.get_disease_info(predicted_class)
        
        self.results_text.insert(tk.END, "\n🔬 Disease Information:\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        self.results_text.insert(tk.END, f"Description: {disease_info['description']}\n\n")
        self.results_text.insert(tk.END, f"Symptoms: {disease_info['symptoms']}\n\n")
        self.results_text.insert(tk.END, f"Treatment: {disease_info['treatment']}\n")
        
        # Confidence level interpretation
        self.results_text.insert(tk.END, "\n📋 Confidence Interpretation:\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        
        if confidence > 0.8:
            self.results_text.insert(tk.END, "✅ High confidence - Very likely correct\n")
        elif confidence > 0.6:
            self.results_text.insert(tk.END, "⚠️ Moderate confidence - Likely correct\n")
        else:
            self.results_text.insert(tk.END, "❗ Low confidence - Consider expert consultation\n")
    
    def plot_confidence_chart(self, result):
        """Plot confidence chart"""
        self.ax.clear()
        
        # Get top 5 predictions for chart
        all_preds = result['all_predictions']
        sorted_preds = sorted(all_preds.items(), key=lambda x: x[1], reverse=True)[:5]
        
        classes = [item[0].replace('_', ' ') for item in sorted_preds]
        confidences = [item[1] * 100 for item in sorted_preds]
        
        # Create horizontal bar chart
        bars = self.ax.barh(classes, confidences, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0'])
        
        # Customize chart
        self.ax.set_xlabel('Confidence (%)')
        self.ax.set_title('Prediction Confidence')
        self.ax.set_xlim(0, 100)
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            width = bar.get_width()
            self.ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                        f'{conf:.1f}%', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        self.chart_canvas.draw()
    
    def clear_results(self):
        """Clear all results and reset interface"""
        self.current_image_path = None
        self.image_label.configure(image="", text="No image selected")
        self.image_label.image = None
        self.results_text.delete(1.0, tk.END)
        self.predict_btn.configure(state=tk.DISABLED)
        
        # Clear chart
        self.ax.clear()
        self.chart_canvas.draw()
        
        self.status_var.set("Ready")

def main():
    # Check if model exists
    if not os.path.exists('models/best_plant_disease_model.h5'):
        print("Trained model not found!")
        print("Please run 'python model_training.py' first to train the model.")
        
        # Create a simple message window
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        result = messagebox.askyesno(
            "Model Not Found", 
            "The trained model was not found. Would you like to:\n\n"
            "YES: Continue with the GUI (for demo purposes)\n"
            "NO: Exit and train the model first\n\n"
            "Note: Predictions will not work without a trained model."
        )
        
        if not result:
            print("Please run 'python model_training.py' to train the model first.")
            return
        
        root.destroy()
    
    # Create and run GUI
    root = tk.Tk()
    app = PlantDiseaseGUI(root)
    
    # Configure styles
    style = ttk.Style()
    style.theme_use('clam')
    
    print("Plant Disease Detection GUI started!")
    print("Upload a plant leaf image and click 'Predict Disease' to get results.")
    
    root.mainloop()

if __name__ == "__main__":
    main()
