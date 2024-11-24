import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

def detect_objects(image):
    """
    Detect objects in the provided image using DETR model.
    """
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    return results

def draw_boxes(image, results):
    """
    Draw bounding boxes on the image based on DETR results.
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        
        # Draw the bounding box
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label and score
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
        ax.text(xmin, ymin - 10, label_text, color='red', fontsize=12, backgroundcolor='white')
    
    plt.axis('off')
    st.pyplot(fig)

# Streamlit application
st.title("Object Detection")
st.write("This application uses the DETR model to detect objects in an image.")

# Input for image URL
image_url = st.text_input("Enter your image URL:", value="https://i.redd.it/i-asked-ai-to-make-some-vermont-themed-wheres-waldo-v0-oa1ph5rdoi0c1.jpg?width=1024&format=pjpg&auto=webp&s=e047934b3c2058d454296238b5914c79a70380dd")

if image_url:
    try:
        # Load the image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Display the image
        st.image(image, caption="Input Image", use_container_width=True)
        
        # Detect objects
        with st.spinner("Detecting objects..."):
            results = detect_objects(image)
        
        # Display results
        st.write("### Detected Objects")
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            st.write(
                f"Detected **{model.config.id2label[label.item()]}** with confidence "
                f"**{round(score.item(), 3)}** at location {box}"
            )
        
        # Draw and display bounding boxes
        draw_boxes(image, results)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
