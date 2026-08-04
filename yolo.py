from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(
    source="1.jpg",
    conf=0.65,
    save=True
)
