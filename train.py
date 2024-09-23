from ultralytics import YOLO
model = YOLO('yolov8n.yaml')
# Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8n.pt')
# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data='model.yaml', epochs=20)
# Evaluate the model's performance on the validation set
results = model.val()
# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')
# Export the model to ONNX format
success = model.export(format='onnx')