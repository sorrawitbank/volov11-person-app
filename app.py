import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLO Image Detection App :)")

# Load YOLO model
# model = YOLO("runs/detect/train73/weights/best.pt")
model = YOLO("weights/yolo11n.pt")

# Upload image
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

def count_items_dict(items_index_list):
  count_items = {}
  for item in items_index_list:
    if model.names[item] not in count_items:
      count_items[model.names[item]] = 1
    else:
      count_items[model.names[item]] += 1
  return count_items

if uploaded_image is not None:
	# Show original image
	st.image(uploaded_image , caption="Uploaded Image", use_container_width=True)

	# Read image and convert to numpy array
	image = Image.open(uploaded_image)
	image_np = np.array(image)

	# Run YOLO inference
	st.info("Running YOLO object detection...")
	results = model.predict(image_np , conf=0.4)

	# Draw results on image
	result_image = results[0].plot()
	st.image(result_image , caption="YOLO Detection Result", use_container_width=True)
	st.success("Detection completed!")

	# Extract detection results
	items_index = results[0].boxes.cls.cpu().numpy().astype(int)
	count_items = count_items_dict(items_index)

	# Count item
	for item, count in count_items.items() :
		st.write(f"Number of {item} detected: **{count}**")
