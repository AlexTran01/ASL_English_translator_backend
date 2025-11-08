import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk, ImageDraw
import requests
import threading

class SignLanguageTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a1a")
        
        # Variables to manage camera
        self.cap = None
        self.is_camera_running = False
        self.current_frame = None
        # Image
        # image= Image.open("glasses.jpg")
        # photo= ImageTk.PhotoImage(image)
        # label= tk.Label(root,image=photo)
        # label.pack()
        
        

        # Create UI
        self.create_ui()
        
    def create_ui(self):
        """Create user interface with dark theme"""
        
        # Main frame containing camera and text
        main_frame = tk.Frame(self.root, bg="#1a1a1a") 
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Left frame - Camera with Canvas for rounded corners
        camera_container = tk.Frame(main_frame, bg="#1a1a1a")
        camera_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        # Canvas for camera with rounded corners
        self.camera_canvas = tk.Canvas(
            camera_container,
            bg="#2b2b2b",
            highlightthickness=0,
            relief=tk.FLAT
        )
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Draw rounded frame for camera
        self.draw_rounded_rect(self.camera_canvas, 0, 0, 0, 0, "#3a3a3a", 15)
        
        # Label to display video (will be placed in canvas)
        self.video_label = tk.Label(
            self.camera_canvas,
            text="Camera feed will appear here",
            bg="#3a3a3a",
            fg="#cccccc",
            font=("Arial", 14),
            anchor=tk.CENTER
        )
        
        # Red dot indicator (will show when camera is running)
        self.recording_indicator = self.camera_canvas.create_oval(
            20, 20, 35, 35,
            fill="#ff0000",
            outline="",
            state=tk.HIDDEN
        )
        
        # Right frame - Translation result with Canvas for rounded corners
        result_container = tk.Frame(main_frame, bg="#1a1a1a")
        result_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0))
        
        # Canvas for result with rounded corners
        self.result_canvas = tk.Canvas(
            result_container,
            bg="#2b2b2b",
            highlightthickness=0,
            relief=tk.FLAT
        )
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Draw rounded frame for result (with darker background)
        self.draw_rounded_rect(self.result_canvas, 0, 0, 0, 0, "#2d3436", 15)
        
        # Text area to display results (large font, easy to read, beautiful design, no scrollbar)
        self.result_text = tk.Text(
            self.result_canvas,
            font=("Arial", 18),
            bg="#2d3436",
            fg="#ffffff",
            wrap=tk.WORD,
            insertbackground="#00bcd4",
            selectbackground="#00bcd4",
            selectforeground="#ffffff",
            relief=tk.FLAT,
            borderwidth=2,
            highlightthickness=2,
            highlightbackground="#00bcd4",
            highlightcolor="#00bcd4",
            padx=25,
            pady=25
        )
        
        # Frame containing circular arrow button in the center
        center_frame = tk.Frame(main_frame, bg="#1a1a1a")
        center_frame.pack(side=tk.LEFT, padx=10)
        
        # Translate button with circular arrow symbol (blue color)
        self.translate_button = tk.Button(
            center_frame,
            text="⇄",
            font=("Arial", 32, "bold"),
            bg="#00bcd4",
            fg="#ffffff",
            activebackground="#00acc1",
            activeforeground="#ffffff",
            width=3,
            height=2,
            relief=tk.FLAT,
            command=self.translate_sign,
            cursor="hand2",
            bd=0,
            highlightthickness=0
        )
        self.translate_button.pack(pady=50)
        
        # Bind canvas resize to update video and text positions
        self.camera_canvas.bind("<Configure>", self.on_camera_canvas_resize)
        self.result_canvas.bind("<Configure>", self.on_result_canvas_resize)
        
        # Status label (hidden, only shown when needed)
        self.status_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 11),
            bg="#1a1a1a",
            fg="#888888"
        )
        self.status_label.pack(pady=(0, 15))
        
        # Frame containing control buttons at the bottom
        control_frame = tk.Frame(self.root, bg="#1a1a1a")
        control_frame.pack(pady=(0, 20))
        
        # Button to turn camera on/off
        self.camera_button = tk.Button(
            control_frame,
            text="Start Camera",
            font=("Arial", 12),
            bg="#4a4a4a",
            fg="white",
            activebackground="#5a5a5a",
            activeforeground="white",
            padx=25,
            pady=12,
            relief=tk.FLAT,
            command=self.toggle_camera,
            cursor="hand2"
        )
        self.camera_button.pack(side=tk.LEFT, padx=10)
        
        # Button to clear results
        self.clear_button = tk.Button(
            control_frame,
            text="Clear",
            font=("Arial", 12),
            bg="#4a4a4a",
            fg="white",
            activebackground="#5a5a5a",
            activeforeground="white",
            padx=25,
            pady=12,
            relief=tk.FLAT,
            command=self.clear_result,
            cursor="hand2"
        )
        self.clear_button.pack(side=tk.LEFT, padx=10)
    
    def draw_rounded_rect(self, canvas, x1, y1, x2, y2, color, radius):
        """Draw rounded rectangle"""
        width = canvas.winfo_width() if canvas.winfo_width() > 1 else 500
        height = canvas.winfo_height() if canvas.winfo_height() > 1 else 400
        
        # Clear old canvas
        canvas.delete("rounded_rect")
        
        # Draw rectangle with rounded corners (simple approach)
        # Draw main rectangle
        canvas.create_rectangle(
            radius, 0,
            width - radius, height,
            fill=color, outline=color, tags="rounded_rect", width=0
        )
        canvas.create_rectangle(
            0, radius,
            width, height - radius,
            fill=color, outline=color, tags="rounded_rect", width=0
        )
        # Draw rounded corners
        canvas.create_oval(0, 0, radius*2, radius*2, fill=color, outline=color, tags="rounded_rect", width=0)
        canvas.create_oval(width - radius*2, 0, width, radius*2, fill=color, outline=color, tags="rounded_rect", width=0)
        canvas.create_oval(0, height - radius*2, radius*2, height, fill=color, outline=color, tags="rounded_rect", width=0)
        canvas.create_oval(width - radius*2, height - radius*2, width, height, fill=color, outline=color, tags="rounded_rect", width=0)
    
    def on_camera_canvas_resize(self, event):
        """Update when camera canvas changes size"""
        self.draw_rounded_rect(self.camera_canvas, 0, 0, event.width, event.height, "#3a3a3a", 15)
        # Update video label position
        self.camera_canvas.update_idletasks()
        if self.video_label.winfo_ismapped():
            self.camera_canvas.coords(self.video_label.winfo_id(), event.width // 2, event.height // 2)
    
    def on_result_canvas_resize(self, event):
        """Update when result canvas changes size"""
        # Draw rounded frame with blue accent
        self.draw_rounded_rect(self.result_canvas, 0, 0, event.width, event.height, "#2d3436", 15)
        # Update result_text position and size
        if hasattr(self, 'result_text_window'):
            self.result_canvas.coords(
                self.result_text_window,
                event.width // 2,
                event.height // 2
            )
            self.result_canvas.itemconfig(
                self.result_text_window,
                width=event.width - 40,
                height=event.height - 40
            )
        else:
            # Create window for result_text for the first time
            self.result_text_window = self.result_canvas.create_window(
                event.width // 2,
                event.height // 2,
                window=self.result_text,
                anchor=tk.CENTER,
                width=event.width - 40,
                height=event.height - 40
            )
        
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_label.config(text="Cannot open camera", fg="#ff4444")
                return
            
            self.is_camera_running = True
            self.camera_button.config(text="Stop Camera", bg="#ff4444")
            # Show red dot indicator
            self.camera_canvas.itemconfig(self.recording_indicator, state=tk.NORMAL)
            self.status_label.config(text="Camera is running...", fg="#44ff44")
            # Place video label in canvas
            self.video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            self.update_video()
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="#ff4444")
    
    def stop_camera(self):
        """Stop camera"""
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
        self.camera_button.config(text="Start Camera", bg="#4a4a4a")
        # Hide red dot indicator
        self.camera_canvas.itemconfig(self.recording_indicator, state=tk.HIDDEN)
        self.video_label.config(image='', text="Camera stopped")
        self.video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.status_label.config(text="Camera stopped", fg="#888888")
    
    def update_video(self):
        """Update video frame"""
        if self.is_camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Convert color from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get canvas size
                canvas_width = self.camera_canvas.winfo_width()
                canvas_height = self.camera_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    # Calculate size to maintain aspect ratio
                    frame_height, frame_width = frame.shape[:2]
                    scale = min((canvas_width - 40) / frame_width, (canvas_height - 40) / frame_height)
                    new_width = int(frame_width * scale)
                    new_height = int(frame_height * scale)
                    
                    # Resize frame
                    frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert to ImageTk for display
                    image = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(image=image)
                    
                    self.video_label.config(image=photo, text="", bg="#3a3a3a")
                    self.video_label.image = photo  # Keep reference
                    self.current_frame = frame
                    # Update label position
                    self.video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                
        if self.is_camera_running:
            # Call again after 30ms to update next frame
            self.root.after(30, self.update_video)
    
    def translate_sign(self):
        """Send request to API for translation"""
        if not self.is_camera_running or self.current_frame is None:
            self.status_label.config(text="Please start camera first!", fg="#ff4444")
            return
        
        self.status_label.config(text="Translating...", fg="#ffaa00")
        self.translate_button.config(state=tk.DISABLED, bg="#00838f")
        
        # Run in separate thread to not block UI
        thread = threading.Thread(target=self.call_api)
        thread.daemon = True
        thread.start()
    
    def call_api(self):
        """Call backend API (simulated or real)"""
        try:
            # TODO: Change this URL to your actual API
            api_url = "http://localhost:5000/api/translate"
            
            # Simulate response (remove this line when you have real API)
            # response = requests.post(api_url, json={"frame": "..."}, timeout=5)
            
            # Simulate translation result (remove when you have real API)
            import time
            time.sleep(1)  # Simulate delay
            translated_text = "Hello! (Simulated result - needs real API connection)"
            
            # Update UI in main thread
            self.root.after(0, self.update_result, translated_text)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API connection error: {str(e)}"
            self.root.after(0, self.update_result, error_msg)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, self.update_result, error_msg)
    
    def update_result(self, text):
        """Update translation result on UI"""
        self.result_text.insert(tk.END, text + "\n\n")
        self.result_text.see(tk.END)
        self.status_label.config(text="Translation complete!", fg="#44ff44")
        self.translate_button.config(state=tk.NORMAL, bg="#00bcd4")
    
    def clear_result(self):
        """Clear results"""
        self.result_text.delete(1.0, tk.END)
        self.status_label.config(text="Cleared", fg="#888888")
    
    def __del__(self):
        """Cleanup when closing application"""
        if self.cap:
            self.cap.release()

def main():
    root = tk.Tk()
    app = SignLanguageTranslator(root)
    root.mainloop()

if __name__ == "__main__":
    main()

