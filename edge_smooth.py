# The edge_smooth.py is from taki0112/CartoonGAN-Tensorflow https://github.com/taki0112/CartoonGAN-Tensorflow#2-do-edge_smooth
from tools.utils import check_folder
import numpy as np
import cv2, os, argparse
from glob import glob
from tqdm import tqdm

def parse_args():
    desc = "Edge smoothed"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Shinkai', help='dataset_name')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')

    return parser.parse_args()

def make_edge_smooth(dataset_name, img_size) :
    check_folder('./dataset/{}/{}'.format(dataset_name, 'smooth'))

    file_list = glob('./dataset/{}/{}/*.*'.format(dataset_name, 'style'))
    save_dir = './dataset/{}/smooth'.format(dataset_name)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    for f in tqdm(file_list) :
        file_name = os.path.basename(f)

        bgr_img = cv2.imread(f)
        gray_img = cv2.imread(f, 0)

        bgr_img = cv2.resize(bgr_img, (img_size, img_size))
        pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (img_size, img_size))

        edges = cv2.Canny(gray_img, 100, 200)      
        dilation = cv2.dilate(edges, kernel)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

        gauss_img = np.copy(bgr_img)
        idx = np.where(closing != 0)
        for i in range(np.sum(closing != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum( np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum( np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum( np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))


         bilateral = cv2.bilateralFilter( bgr_img, 5, 25, 25 )

        # Increase Saturation of Edges
        # kernel_sharpening = np.array([[-1, -1, -1], [-1, 9,-1], [-1, -1, -1]])
        # sharpen = cv2.filter2D(bilateral, -1, kernel_sharpening)

        # Remove Noise From Sharpen Image
        # denoise = cv2.fastNlMeansDenoisingColored(sharpen, None, 10, 10, 7, 15)

        # cv2.imshow( 'sharpen', sharpen )
        # cv2.imshow( 'bgr_img', bgr_img )
        # cv2.waitKey(0)

        # cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)
        cv2.imwrite(os.path.join(save_dir, file_name), bilateral)

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    make_edge_smooth(args.dataset, args.img_size)


if __name__ == '__main__':
    main()
