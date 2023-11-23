import ast
import cv2
import torch
import os
import math
import numpy as np
from image_recognition import preparedata
from ultralytics import YOLO

# Get specific frame of video
def get_frame(framenum, file):
    cap = cv2.VideoCapture(file)
    ret, frame = cap.read()
    i = 0
    while ret:
        if i == framenum:
            cv2.imshow(f"Frame {framenum}", frame)
            cv2.waitKey(0)
        ret, frame = cap.read()
        i+=1
 
# Show a video
def saveVideo(videopath, weightspath):
    video_path_out = '{}_out.mp4'.format(videopath)
    cap = cv2.VideoCapture(videopath)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
    # Load a model
    model = YOLO(weightspath)  # load a custom model
    threshold = 0.1
    while ret:
        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
        out.write(frame)
        ret, frame = cap.read()
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
# Load all the data and return both the framelist and the modellist
def load_data(file): # Input entire path
    filename = file.split('/')[-1].split('.')[0]

    # Load all the frames
    framelist = [(i, file) for i, file in enumerate(os.listdir(f"data/temp/npdata/{filename}"))]

    # Load the model results
    with open(f'data/temp/outdata/{filename}.txt') as fin:
        width, height, fps = map(float, fin.readline().split(','))
        modellist = ast.literal_eval(fin.readline())

    return framelist, modellist, int(width), int(height), fps

# Analyze frames to find the one of interest
def find_point_of_contact(modellist):
    mindist = 999999999999999
    framenum = 0
    for i, nlist in modellist:
        if nlist[0] is None:
            continue
        if len(nlist) == 2 and nlist[0][-1] != nlist[1][-1]:
            cx_1 = (nlist[0][0] + nlist[0][2]) / 2
            cy_1 = (nlist[0][1] + nlist[0][3]) / 2

            cx_2 = (nlist[1][0] + nlist[1][2]) / 2
            cy_2 = (nlist[1][1] + nlist[1][3]) / 2
            prevmindist = mindist
            mindist = min(mindist, math.dist((cx_1, cy_1), (cx_2, cy_2)))
            # print(i, math.dist((cx_1, cy_1), (cx_2, cy_2)), framenum, mindist)
            if prevmindist > mindist:
                framenum = i
    return framenum

# Add blank frames to start of video
def add_blank_frames(n, filename, outname, width, height, fps):
    # Create blank frames
    imarray = [(-1, np.zeros((int(height), int(width), 3), np.uint8)) for i in range(n)]
    for file in os.listdir(f'data/temp/npdata/{filename}'):
        img = cv2.imread(f'data/temp/npdata/{filename}/{file}')
        try:
            imarray.append((int(file.split('.')[0]), img))
        except ValueError:
            continue
    # print(img.shape, np.zeros((int(height), int(width), 3), np.uint8).shape)
    imarray.sort(key=lambda x: x[0])
    out = cv2.VideoWriter(f'data/temp/{outname}', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i,f in imarray:
        out.write(f)
    out.release()

# Trims unnecessary bits and converts to video in temp
def trimFiles(filename, mid, left, right, outname, width, height, fps):
    framelist = []
    # Select necessary and delete unnecessary frames
    for file in os.listdir(f'data/temp/npdata/{filename}'):
        try:
            if mid-left <= int(file.split('.')[0]) <= mid+right:
                framelist.append((int(file.split('.')[0]), cv2.imread(f'data/temp/npdata/{filename}/{file}')))
        except ValueError:
            continue

    # Write the frames
    if outname in os.listdir('data/temp'):
        os.remove(f"data/temp/{outname}")
    # print("LEN FRAMELIST", len(framelist))
    framelist.sort(key=lambda x: x[0])
    out = cv2.VideoWriter(f'data/temp/{outname}', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for i,f in framelist:
        out.write(f)
    out.release()

# Output two horizontally stacked videos
def lineTwoVids(path1, path2, output_name): # output.mp4
    # Get file name without .mp4
    file1 = path1.split('/')[-1].split('.')[0]
    file2 = path2.split('/')[-1].split('.')[0]

    # Load data for both videos
    framelist_1, modellist_1, width1, height1, fps1 = load_data(path1)
    framelist_2, modellist_2, width2, height2, fps2 = load_data(path2)

    # Find point of contact for both videos
    pc1 = find_point_of_contact(modellist_1)
    pc2 = find_point_of_contact(modellist_2)

    # Trim both videos according to their pcs and make into video
    trimFiles(file1, pc1, 60, 60, 'vid1.mp4', width1, height1, fps1)
    trimFiles(file2, pc2, 60, 60, 'vid2.mp4', width2, height2, fps2)

    # Combine both videos
    if output_name in os.listdir('data/output'):
        os.remove(f'data/output/{output_name}')
    # ffmpeg.filter([video1, video2], 'hstack').output(output_name, format='mp4').run() too slow
    os.system(f"ffmpeg -i data/temp/vid1.mp4 -i data/temp/vid2.mp4 -filter_complex hstack data/output/{output_name}")
    # print('done')
    
    
def getSwingFromVideo(path1, output_name):
    preparedata(path1)
    filename = path1.split('/')[-1].split(".")[0] # Get the filename without extension
    print("Filename: ", filename)
    # Load data for video and find the point of contact
    frameList, modelList, width, height, fps = load_data(path1)
    contactPoint = find_point_of_contact(modelList)
    trimFiles(filename, contactPoint, 2*fps, 2*fps, f'{output_name}.mp4', width, height, fps)
    print("Done")

# getSwingFromVideo("videos/Untitled.mp4", "test_output_2")