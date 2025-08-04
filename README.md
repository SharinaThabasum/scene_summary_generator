# scene_summary_generator , an ASSISTIVE TECH for impaired vision people 
A real-time computer vision application that captures live video feed from a mobile camera (via DroidCam) using OpenCV and generates natural language descriptions of the scene using a pre-trained Generative AI model (Hugging Face Transformers).

Features:
📸 Captures real-time video feed using mobile phone camera
👀 Detects frames and extracts key scene content
🧠 Generates human-readable natural language summaries using Generative AI
✅ Supports both USB and Wi-Fi mobile camera connections via DroidCam
💬 Prints scene summaries in real-time to the terminal.

Tech Stack:
Python
OpenCV – Video capture and frame processing
Hugging Face Transformers – For image-to-text scene description (e.g., BLIP/ViT-GPT2)
Torch / TensorFlow – Backend for AI models
DroidCam – Mobile camera streaming to PC

Requirements:
pip install opencv-python transformers torch torchvision

Mobile Setup (DroidCam): 
Install DroidCam on both your Android device and PC.
Enable Developer Options and USB Debugging on your mobile.
Connect via USB or Wi-Fi (IP address will be shown on the DroidCam app).
Use the video stream link (e.g., http://192.168.1.xxx:4747/video) in your Python script.

How It Works: 
The webcam feed is captured using cv2.VideoCapture.
Every few frames, a snapshot is passed to a pre-trained Vision-to-Text model.
The model returns a natural language summary of the frame.
Summary is displayed in the terminal (can be extended to overlay on video, store logs, etc.)

sample output: 
Scene Summary: A group of people are walking down a busy street at night.
Scene Summary: A lone woman standing in a dimly lit area.
