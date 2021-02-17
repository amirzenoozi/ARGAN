from numpy.lib.type_check import real
from tools.frechet_kernel_Inception_distance import *
from tools.inception_score import *
from skimage.measure import compare_ssim
from glob import glob
import os
import argparse
import warnings
import numpy as np
import cv2

def parse_args():
    desc = "Edge smoothed"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--directory', type=str, default='evaluation', help='Evaluation Folder Name')

    return parser.parse_args()

def inception_score(args):
    filenames = glob(os.path.join(f'{ args.directory }/fake', '*.*'))
    images = [get_images(filename) for filename in filenames]
    images = np.transpose(images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])

    logits = inception_logits(inception_images)

    IS = get_inception_score(BATCH_SIZE, images, inception_images, logits, splits=10)

    print("===========================")
    print("IS : ", IS)

def frechet_inception_distance(args) :
    filenames = glob(os.path.join(f'{ args.directory }/real_target', '*.*'))
    real_images = [get_images(filename) for filename in filenames]
    real_images = np.transpose(real_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join(f'{ args.directory }/fake', '*.*'))
    fake_images = [get_images(filename) for filename in filenames]
    fake_images = np.transpose(fake_images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    fcd = frechet_classifier_distance_from_activations(real_activation, fake_activation)
    activations = inception_activations(inception_images)

    FID = get_fid(fcd, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation, activations)

    print("===========================")
    print("FID : ", FID / 100)

def kernel_inception_distance(args) :
    filenames = glob(os.path.join(f'{ args.directory }/real_target', '*.*'))
    real_images = [get_images(filename) for filename in filenames]
    real_images = np.transpose(real_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join(f'{ args.directory }/fake', '*.*'))
    fake_images = [get_images(filename) for filename in filenames]
    fake_images = np.transpose(fake_images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    kcd_mean, kcd_stddev = kernel_classifier_distance_and_std_from_activations(real_activation, fake_activation, max_block_size=10)
    activations = inception_activations(inception_images)

    KID_mean = get_kid(kcd_mean, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation, activations)
    KID_stddev = get_kid(kcd_stddev, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation, activations)

    print("===========================")
    print("KID_mean : ", KID_mean * 100)
    print("KID_stddev : ", KID_stddev * 100)

def mean_kernel_inception_distance() :
    source_alpha = 0.98
    target_alpha = 1 - source_alpha
    
    filenames = glob(os.path.join('./real_source', '*.*'))
    real_source_images = [get_images(filename) for filename in filenames]
    real_source_images = np.transpose(real_source_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join('./real_target', '*.*'))
    real_target_images = [get_images(filename) for filename in filenames]
    real_target_images = np.transpose(real_target_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join('./fake', '*.*'))
    fake_images = [get_images(filename) for filename in filenames]
    fake_images = np.transpose(fake_images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    fcd = frechet_classifier_distance_from_activations(real_activation, fake_activation)
    kcd_mean, kcd_stddev = kernel_classifier_distance_and_std_from_activations(real_activation, fake_activation,
                                                                               max_block_size=10)
    activations = inception_activations(inception_images)

    FID = get_fid(fcd, BATCH_SIZE, real_target_images, fake_images, inception_images, real_activation, fake_activation, activations)
    KID_mean = get_kid(kcd_mean, BATCH_SIZE, real_target_images, fake_images, inception_images, real_activation, fake_activation, activations)
    KID_stddev = get_kid(kcd_stddev, BATCH_SIZE, real_target_images, fake_images, inception_images, real_activation, fake_activation, activations)

    mean_FID = get_fid(fcd, BATCH_SIZE, real_source_images, fake_images, inception_images, real_activation, fake_activation, activations)
    mean_KID_mean = get_kid(kcd_mean, BATCH_SIZE, real_source_images, fake_images, inception_images, real_activation, fake_activation, activations)
    mean_KID_stddev = get_kid(kcd_stddev, BATCH_SIZE, real_source_images, fake_images, inception_images, real_activation, fake_activation, activations)

    mean_FID = (target_alpha * FID + source_alpha * mean_FID) / 2.0
    mean_KID_mean = (target_alpha * KID_mean + source_alpha * mean_KID_mean) / 2.0
    mean_KID_stddev = (target_alpha * KID_stddev + source_alpha * mean_KID_stddev) / 2.0

    # mean_FID = (2 * FID * mean_FID) / (FID + mean_FID)
    # mean_KID_mean = (2 * KID_mean * mean_KID_mean) / (KID_mean + mean_KID_mean)
    # mean_KID_stddev = (2 * KID_stddev * mean_KID_stddev) / (KID_stddev + mean_KID_stddev)
    
    print("===========================")
    print("mean_FID : ", mean_FID / 100)
    print("mean_KID_mean : ", mean_KID_mean * 100)
    print("mean_KID_stddev : ", mean_KID_stddev * 100)

def peak_signalToNoise_ratio( args, max_value=255 ):
    PSNR_list = []

    filenames = glob(os.path.join(f'{ args.directory }/real_target', '*.*'))
    real_images = [get_images(filename) for filename in filenames]
    real_images = np.transpose(real_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join(f'{ args.directory }/fake', '*.*'))
    fake_images = [get_images(filename) for filename in filenames]
    fake_images = np.transpose(fake_images, axes=[0, 3, 1, 2])
    
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    for i in range(len( fake_images )):
        mse = np.mean((np.array(fake_images[i], dtype=np.float32) - np.array(real_images[i], dtype=np.float32)) ** 2)

        if mse == 0:
            PSNR = 100
        PSNR = 20 * np.log10(max_value / (np.sqrt(mse)))

        PSNR_list.append( PSNR )

    print("===========================")
    print( "PSNR_Mean: ", np.mean( PSNR ) )
    
def structural_similarity_index( args ):
    """ If you compare 2 exact images, the value of SSIM should be obviously 1.0 """
    SSIM_list = []

    real_images = glob(os.path.join(f'{ args.directory }/real_target', '*.*'))
    fake_images = glob(os.path.join(f'{ args.directory }/fake', '*.*'))

    for i in range(len( fake_images )):
        # 3. Load the two input images
        imageA = cv2.imread( real_images[i] )
        imageB = cv2.imread( fake_images[i] )

        # 4. Convert the images to grayscale
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        # 5. Compute the Structural Similarity Index (SSIM) between the two
        #    images, ensuring that the difference image is returned
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")

        # 6. You can print only the score if you want
        SSIM_list.append( score )

    print("===========================")
    print("SSIM: ", np.mean(score))

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    inception_score( args )
    frechet_inception_distance( args )
    kernel_inception_distance( args )
    peak_signalToNoise_ratio( args )
    structural_similarity_index( args )


if __name__ == '__main__':
    main()