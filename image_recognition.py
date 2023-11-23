import torch
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
import os
import shutil
from ultralytics import YOLO


# Get model predictions
def get_pandalist(filename):
	model = YOLO("runs/detect/train4/weights/best.pt")
	cap = cv2.VideoCapture(filename)
	ret, frame = cap.read()
	i = 0
	framelist = []
	while ret:
		frame = frame[..., ::-1]
		df = model(frame).pandas().xyxy[0]
		if not df.empty:
			framelist.append([i, frame, df.values.tolist()])
		else:
			framelist.append([i, frame, [None, None]])
		ret, frame = cap.read()
		i += 1
	# print(framelist)
	# returns the model predictions, frame width, height and fps
	print("GOT PANDALIST")
	return framelist, cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)


# Save all the frames from video
def savearray(filename, inlist):
	fname = filename.split('/')[-1].split('.')[0]
	print("FNAME: ", fname)
	try:
		os.mkdir(f'data/temp/npdata/{fname}')
		print("Made First Try")
	except FileExistsError:
		print("Failed first try")
		shutil.rmtree(f'data/temp/npdata/{fname}')
		os.mkdir(f'data/temp/npdata/{fname}')
	print("Saving frames...")
	for i, frame, junk in tqdm(inlist):
		try:
			plt.imsave(f'data/temp/npdata/{fname}/{i}.jpg', frame)
		except FileExistsError:
			os.remove(f'data/temp/npdata/{fname}/{i}.jpg')
			plt.imsave(f'data/temp/npdata/{fname}/{i}.jpg', frame)
		inlist[i].pop(1)
	return inlist


# Unpack mp4 to usable data
def preparedata(infile):
	plist, width, height, fps = get_pandalist(infile)
	plist = savearray(infile.split('.')[0], plist)
	
	fout = open(f'data/temp/outdata/{infile.split("/")[-1].split(".")[0]}.txt', 'w')
	fout.write(f"{width}, {height}, {fps}\n{str(plist)}")
	fout.close()
