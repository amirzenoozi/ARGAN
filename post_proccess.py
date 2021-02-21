from glob import glob
from tqdm import tqdm
from rio_color.operations import parse_operations
from PIL import Image, ImageFilter, ImageEnhance    
from shutil import copyfile

import os
import argparse
import numpy as np
import cv2
import math
import sys

def parse_args():
    desc = "Tensorflow implementation of AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--directory', type=str, default='post_proccess/', help='Evaluation Folder Name')
    parser.add_argument('--color_adj', type=int, default=8, help='Color Value adj. Perameter 8 < X < 20')

    return parser.parse_args()

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix
		
def simplest_cb(img, percent):
    half_percent = percent / 100.0 # depending on the pod a range of 10.0-25.0 returns a pretty good result

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high percentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[int(math.floor(n_cols * half_percent))]
        high_val = flat[int(math.ceil(n_cols * (1.0 - half_percent)))]

        # print("Lowval: ", low_val)
        # print("Highval: ", high_val)

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 200, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

def color_fix( file, perc_val ):
    # img = cv2.imread( file ) # add parameter for image src, example: python color_fix "c:\path\path\path.jpg"
    img = file
    # copyfile(file, file[:-4] + "_original.bmp") #backup_img(img)
    out = simplest_cb(img, perc_val) # value changes intesity 8 to 20 seems to be a good range
    out = np.asarray( out )

    return out
    # cv2.imwrite(file[:-4] + "_fix.png",out)

def frame_enhance( frame ):
    color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray( color_coverted )
    enh = ImageEnhance.Contrast(im)
    enh = enh.enhance(1.5)
    finale_img = cv2.cvtColor( np.asarray( enh ), cv2.COLOR_RGB2BGR)
    
    return finale_img

def read_video_file( video_file_name, args, output_format='MP4V' ):
    vid = cv2.VideoCapture( video_file_name )
    vid_name = os.path.basename( video_file_name )
    total_frames = int( vid.get(cv2.CAP_PROP_FRAME_COUNT) )
    fps = int( vid.get(cv2.CAP_PROP_FPS) )
    codec = cv2.VideoWriter_fourcc(*output_format)
    frame_count = 0


    ret, img = vid.read()
    if img is None:
        print('Error! Failed to determine frame size: frame empty.')
        return
    height, width = img.shape[:2]
    out = cv2.VideoWriter(os.path.join( 'video/output', vid_name), codec, fps, (width, height))

    pbar = tqdm(total=total_frames)
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while( vid.isOpened() ):
        ret, frame = vid.read()
        
        if ret and frame_count <= total_frames:
            # Using Pillow
            out.write( frame_enhance( frame ) )
            # Using OpenCV
            # out.write( color_fix( frame, args.color_adj ) )
            frame_count += 1
            pbar.update(1)
        else:
            break

    pbar.update( total_frames )
    pbar.close()
    vid.release()
    cv2.destroyAllWindows()

def proccess_images( directory ):
    files = glob(os.path.join(f'{ directory }', '*.*'))

    for i in range(len( files )):
        file_name = os.path.basename( files[i] ).split('.')[0]

        image = cv2.imread( files[i] )
        color_coverted = cv2.cvtColor( image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray( color_coverted )
        enh = ImageEnhance.Contrast(im)
        enh = enh.enhance(1.5)
        finale_img = cv2.cvtColor( np.asarray( enh ), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{ directory }/{file_name}_enhance.png', finale_img)

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    read_video_file( 'video/output/Paprika/4-res18_(glr_1e-5).mp4', args )
    # proccess_images( args.directory )