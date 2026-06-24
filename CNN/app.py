import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# ==================================
# Load Trained Model
# ==================================
model = tf.keras.models.load_model("cats_dogs_model.keras")

IMG_SIZE = 200


# ==================================
# Prediction Function
# ==================================
def predict(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)

    # Handle RGBA images
    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = float(model.predict(image, verbose=0)[0][0])

    if prediction > 0.5:
        label = "🐶 Dog"
        confidence = prediction * 100
    else:
        label = "🐱 Cat"
        confidence = (1 - prediction) * 100

    return f"""
🎯 Prediction: {label}

📊 Confidence: {confidence:.2f}%
"""


# ==================================
# UI
# ==================================
with gr.Blocks(title="Cat vs Dog Detector") as demo:

    gr.Markdown("""
# 🐱🐶 Cat vs Dog Detector

### Deep Learning Based Image Classification System

Upload an image and the CNN model will predict whether it contains a **Cat 🐱** or a **Dog 🐶**.

---

### 🚀 Technologies Used

- Python
- TensorFlow
- Keras
- CNN (Convolutional Neural Network)
- Gradio
- Hugging Face Spaces

---

### 📊 Model Performance

**Validation Accuracy: 75.8%**

---

### 👨‍💻 Developed By

**Bhupinder Sandhu**

---

⚠️ **Disclaimer:** This AI model is not 100% accurate and may occasionally misclassify images. Results are intended for educational and demonstration purposes only.
""")

    with gr.Row():

        image_input = gr.Image(
            type="pil",
            label="📷 Upload Cat or Dog Image",
            height=400
        )

        output = gr.Textbox(
            label="🎯 Prediction Result",
            lines=4
        )

    with gr.Row():

        predict_btn = gr.Button(
            "🔍 Predict",
            variant="primary"
        )

        clear_btn = gr.ClearButton(
            [image_input, output],
            value="🗑️ Clear"
        )

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=output
    )

    gr.Markdown("""
---
Made with ❤️ using TensorFlow, Gradio and Hugging Face

© 2026 Bhupinder Sandhu
""")

demo.launch()
