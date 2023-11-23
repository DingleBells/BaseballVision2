from ultralytics import YOLO

model=YOLO("runs/detect/train4/weights/best.pt")

result = model('videos/testFrame.png')
print(result[0])