import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model("efficientnetv2_grape_final.keras")

# ==========================================
# CLASS NAMES
# (Keep same order as training)
# ==========================================

classes = [
    "Black Rot",
    "ESCA",
    "Healthy",
    "Leaf Blight"
]

# ==========================================
# SELECT IMAGE
# ==========================================

root = Tk()
root.withdraw()

img_path = askopenfilename(
    title="Select Grape Leaf Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png")
    ]
)

if img_path == "":
    print("No image selected.")
    exit()

# ==========================================
# LOAD IMAGE
# ==========================================

img = image.load_img(img_path, target_size=(384, 384))

img_array = image.img_to_array(img)

# SAME preprocessing as training
img_array = preprocess_input(img_array)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# ==========================================
# PREDICT
# ==========================================

prediction = model.predict(img_array, verbose=0)

predicted_index = np.argmax(prediction)

predicted_class = classes[predicted_index]

confidence = prediction[0][predicted_index] * 100

# ==========================================
# DISPLAY IMAGE
# ==========================================

plt.figure(figsize=(7,7))
plt.imshow(img)
plt.axis("off")

if confidence >= 70:

    plt.title(
        f"Prediction : {predicted_class}\n"
        f"Confidence : {confidence:.2f}%"
    )

else:

    plt.title(
        f"Prediction : UNKNOWN\n"
        f"Confidence : {confidence:.2f}%"
    )

plt.show()

# ==========================================
# PRINT PROBABILITIES
# ==========================================

print("\n==============================")
print("Prediction Probabilities")
print("==============================")

for i, cls in enumerate(classes):
    print(f"{cls:<15}: {prediction[0][i]*100:.2f}%")

print("==============================")

# ==========================================
# FINAL RESULT
# ==========================================

if confidence >= 70:

    print("\nDisease Detected")
    print("----------------")
    print("Disease   :", predicted_class)
    print(f"Confidence: {confidence:.2f}%")

else:

    print("\nWARNING")
    print("----------------")
    print("Unknown disease or low confidence prediction.")
    print("This image may not belong to any disease")
    print("that the AI model was trained on.")
    print(f"Highest Confidence: {confidence:.2f}%")