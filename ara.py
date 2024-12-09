from ultralytics import YOLO

# Load Backup Training model
# model = YOLO("best.pt")

# Export the model to TFLite format
# model.export(format="tflite")  # creates 'yolo11n_float32.tflite'

# Load the exported TFLite model
tflite_model = YOLO("best_saved_model/best_float32.tflite")

# Run inference
results = tflite_model("car.jpg")