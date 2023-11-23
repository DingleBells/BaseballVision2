from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('runs/detect/train4/weights/best.pt')


# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='/Users/kangheecho/PycharmProjects/datasets/data.yaml', epochs=100,device='mps')

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model('/Users/kangheecho/PycharmProjects/baseballvision2/videos/testVideo.mov')

model.export(format='onnx', dynamic=True)