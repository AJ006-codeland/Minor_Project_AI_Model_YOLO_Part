# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import cv2
import matplotlib.pyplot as plt

from tkinter import Tk
from tkinter.filedialog import askopenfilename

from ultralytics import YOLO

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input


# ============================================================
# LOAD MODELS
# ============================================================

# YOLOv8 Segmentation model (yolov8n-seg)
yolo_model = YOLO("best.pt")

# EfficientNetV2 Disease Classifier
disease_model = load_model("efficientnetv2_grape_final.keras")

# ============================================================
# CLASS NAMES
# ============================================================

classes = [
    "Black Rot",
    "ESCA",
    "Healthy",
    "Leaf Blight"
]

# Disease confidence threshold
DISEASE_CONFIDENCE_THRESHOLD = 40

# YOLO detection confidence
YOLO_CONF = 0.65

# Background fill mode for masked leaf: "black" or "white"
MASK_BACKGROUND = "black"


# ============================================================
# COLORS
# ============================================================

colors = {
    "Healthy": (0, 255, 0),
    "Black Rot": (255, 0, 0),
    "ESCA": (255, 165, 0),
    "Leaf Blight": (255, 255, 0),
    "Unknown Disease": (180, 180, 180)
}

# ============================================================
# SELECT IMAGE
# ============================================================

root = Tk()
root.withdraw()

img_path = askopenfilename(
    title="Select a grape leaf image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp *.jfif *.webp")
    ]
)

if img_path == "":
    print("No image selected.")
    exit()


# ============================================================
# LOAD IMAGE
# ============================================================

original_image = cv2.imread(img_path)

if original_image is None:
    print("Error: Unable to read image.")
    exit()

original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

height, width = original_image_rgb.shape[:2]


# ============================================================
# YOLOv8 SEGMENTATION
# ============================================================

results = yolo_model.predict(
    source=original_image,  # original_image_rgb
    conf=YOLO_CONF,
    save=True,  # False
    verbose=False
)

result = results[0]

boxes = result.boxes
masks = result.masks  # None if model is detection-only, not seg

# ============================================================
# CHECK IF ANY GRAPE LEAF EXISTS
# ============================================================

if boxes is None or len(boxes) == 0:

    print("\n==============================")
    print("FINAL RESULT")
    print("==============================")
    print("Status : UNKNOWN PLANT")
    print("==============================")

    plt.figure(figsize=(8, 8))
    plt.imshow(original_image_rgb)
    plt.axis("off")
    plt.title("UNKNOWN PLANT")
    plt.show()

    exit()

print(f"\nDetected {len(boxes)} grape leaf/leaves.\n")

# ============================================================
# PRE-RESIZE ALL MASKS TO ORIGINAL IMAGE SIZE (if available)
# ============================================================

if masks is not None:
    # masks.data shape: (N, mask_h, mask_w) at model's internal resolution
    masks_np = masks.data.cpu().numpy()
else:
    masks_np = None
    print("WARNING: No masks found on this result — falling back to plain "
          "bounding-box crops (background will NOT be removed). "
          "Make sure best.pt is a -seg model if you expected masking.")


# ============================================================
# PROCESS EACH DETECTED LEAF
# ============================================================

annotated_image = original_image_rgb.copy()

diagnosis_labels = []

for i in range(len(boxes)):

    # -----------------------------
    # Bounding Box (with padding)
    # -----------------------------
    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

    pad = 10

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)

    # -----------------------------
    # Build a leaf-only crop (mask background removed)
    # -----------------------------
    if masks_np is not None:

        # Resize this leaf's mask up to the original image resolution
        mask_full = cv2.resize(
            masks_np[i],
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )
        mask_bin = (mask_full > 0.5).astype(np.uint8)

        # Apply mask to the full image first, then crop to the box.
        # This ensures pixels from overlapping/neighboring leaves and
        # background clutter are zeroed out before EfficientNet sees them.
        fill_value = 0 if MASK_BACKGROUND == "black" else 255

        masked_full = original_image_rgb.copy()
        masked_full[mask_bin == 0] = fill_value

        crop = masked_full[y1:y2, x1:x2]

    else:
        # Fallback: plain bbox crop (old behavior, no masking)
        crop = original_image_rgb[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    # -----------------------------
    # Prepare image for EfficientNet
    # -----------------------------
    crop_resized = cv2.resize(crop, (384, 384))

    img = crop_resized.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = disease_model.predict(img, verbose=0)[0]

    predicted_index = np.argmax(prediction)

    predicted_class = classes[predicted_index]

    confidence = prediction[predicted_index] * 100

    # -----------------------------
    # Final Decision
    # -----------------------------
    if confidence >= DISEASE_CONFIDENCE_THRESHOLD:
        final_label = predicted_class
    else:
        final_label = "Unknown Disease"

    diagnosis_labels.append(final_label)

    box_color = colors.get(final_label, (255, 255, 255))

    # -----------------------------
    # Draw Segmentation Outline (if mask available)
    # -----------------------------
    if masks_np is not None:
        contours, _ = cv2.findContours(
            mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated_image, contours, -1, box_color, 2)

    # -----------------------------
    # Draw Bounding Box
    # -----------------------------
    cv2.rectangle(
        annotated_image,
        (x1, y1),
        (x2, y2),
        box_color,
        2
    )

    label = f"{final_label} ({confidence:.1f}%)"

    cv2.putText(
        annotated_image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        box_color,
        2
    )

    # -----------------------------
    # Console Output
    # -----------------------------
    print(f"\n========== Leaf {i+1} ==========")

    for j, cls in enumerate(classes):
        print(f"{cls:<15}: {prediction[j]*100:.2f}%")

    print("-------------------------------")
    print(f"Prediction : {final_label}")
    print(f"Confidence : {confidence:.2f}%")

# ============================================================
# FINAL DIAGNOSIS
# ============================================================

if len(diagnosis_labels) == 0:

    overall_result = "No grape leaf detected"

elif len(set(diagnosis_labels)) == 1:

    overall_result = diagnosis_labels[0]

else:

    counts = {}
    for label in diagnosis_labels:
        counts[label] = counts.get(label, 0) + 1

    overall_result = max(counts, key=counts.get)


print("\n========================================")
print("FINAL RESULT")
print("========================================")
print(f"Detected Leaves : {len(diagnosis_labels)}")
print(f"Diagnosis       : {overall_result}")
print("========================================")


# ============================================================
# DISPLAY IMAGE
# ============================================================

plt.figure(figsize=(10, 10))
plt.imshow(annotated_image)
plt.axis("off")
plt.title(f"Diagnosis: {overall_result}")
plt.tight_layout()
plt.show()