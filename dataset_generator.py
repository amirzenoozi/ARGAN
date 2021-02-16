from glob import glob
from tqdm import tqdm
from PIL import Image
    
import cv2
import os
import argparse

def parse_args():
    desc = "Frame Extraction"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--directory', type=str, default='video', help='Video Directory Target')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--file_format', type=str, default='mp4', help='The Files Format')
    parser.add_argument('--skip_frame', type=int, default=90, help='Skip Frame Number')

    return parser.parse_args()

def video_file_list( directory, file_format ):
    video_files = []
    for root, dirs, files in os.walk( os.path.join(os.path.abspath('.'), f"{directory}") ):
        for file in files:
            if(file.endswith( f".{file_format}")):
                video_files.append( os.path.join(root, file) )
    
    return video_files

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    
    return pil_img.crop(((img_width - crop_width) // 2, (img_height - crop_height) // 2, (img_width + crop_width) // 2, (img_height + crop_height) // 2))

def keep_aspect_ratio( pil_img, base_size ):
    wpercent = (base_size/float(pil_img.size[0]))
    hsize = int((float(pil_img.size[1])*float(wpercent)))
    
    return pil_img.resize((base_size,hsize), Image.ANTIALIAS)

def resize( file_path ):
    path = os.path.abspath( file_path.split('/')[0] )
    item = file_path.split('/')[1]
    
    if os.path.isfile( f"{path}/{item}" ):
        im = Image.open( f"{path}/{item}" )
        resize_im = keep_aspect_ratio( im, 600 )
        im_new = crop_center( resize_im, 256, 256 )
        f, e = os.path.splitext( f"{path}/{item}")
        im_new.save( f + '.jpg', 'JPEG', quality=90)

def frame_extraction( video_file_name, file_number, skip_frame ):
    cap = cv2.VideoCapture( video_file_name )
    total_frames = int( cap.get(cv2.CAP_PROP_FRAME_COUNT) )
    frame_rate = skip_frame
    frame_count = 0

    pbar = tqdm(total=total_frames)
    while( cap.isOpened() ):
        ret, frame = cap.read()
        
        if ret and frame_count <= total_frames:
            cv2.imwrite( f"frame/{file_number}.jpg", frame)
            resize( f"frame/{file_number}.jpg" )
            file_number += 1
            frame_count += frame_rate # i.e. at 30 fps, this advances one second
            pbar.update( frame_rate )
            cap.set(1, frame_count) 
        else:
            break

    pbar.update( total_frames )
    pbar.close()
    cap.release()
    cv2.destroyAllWindows()
    
    return file_number

def proccess_files( files_list, args ):
    file_number = 0
    for file in files_list:
        file_number = frame_extraction( file, file_number, args.skip_frame )

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    target_directory_pathes = video_file_list( args.directory, args.file_format )
    proccess_files( target_directory_pathes, args )


if __name__ == '__main__':
    main()