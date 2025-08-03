import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR


# Load the image captioning model and processor from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")  # Use 'cuda' if GPU available

# Connect to your mobile camera (try 1 or 2 based on your system)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_interval = 5  # seconds
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Show the live feed
    cv2.imshow('Live Feed', frame)

    # Every few seconds, capture and summarize
    if time.time() - last_time > frame_interval:
        last_time = time.time()

        # Convert OpenCV image (BGR) to PIL format (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Generate caption
        inputs = processor(pil_image, return_tensors="pt").to("cpu")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)

        print("\n📝 Scene Summary:", description)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
