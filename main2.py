from application import lineTwoVids
from image_recognition import preparedata

def main(vid1_path, vid2_path, output_file_name, reload_images=True):
    """
    vid1, vid2: path/to/vid.mp4
    output_file_name: path/to/output.mp4
    """
    if reload_images:
        preparedata(vid1_path)
        preparedata(vid2_path)
    lineTwoVids(vid1_path, vid2_path, output_file_name)
    
    
if __name__ == '__main__':
    main(input("Path to Video 1?"), input("Path to Video 2?"), input("Output Name?"))