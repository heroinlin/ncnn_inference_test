import os
import cv2
import numpy as np
import argparse


class ImageToNpy():
    def __init__(self, image_path, npy_path):
        super(ImageToNpy, self).__init__()
        self.image_path = image_path
        self.npy_path = npy_path
        self.config = {'width': 224,
                       'height': 224,
                       'mean': [0.4914, 0.4822, 0.4465],
                       'stddev': [0.247, 0.243, 0.261],
                       'divisor': 255.0,
                       'color_format': 'RGB',
                       }

    def set_config(self, key: str, value):
        if key not in self.config:
            print("config key error! please check it!")
            exit()
        self.config[key] = value
    
    def image2npy(self):
        if not os.path.exists(self.image_path):
            print("图像路径错误，请检查！")
            exit(-1)
        image = cv2.imread(self.image_path)
        if self.config['color_format'] == "RGB":
            image = image[:, :, ::-1]
        print(image)
        if self.config['width'] > 0 and self.config['height'] > 0:
            image = cv2.resize(image, (self.config['width'], self.config['height']))
        image = (np.array(image, dtype=np.float32) / self.config['divisor'] - self.config['mean']) / self.config['stddev']
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)
        np.save(self.npy_path, image)
        # image = np.expand_dims(image, 0)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--image_path',
                        type=str,
                        help='image path',
                        default="data/data.png")
    parser.add_argument('-s',
                        '--save_path',
                        type=str,
                        help='npy save path',
                        default="data/data.npy")
    parser.add_argument('--mean',
                        type=float,
                        nargs='+',
                        help='mean value',
                        default=None)
    parser.add_argument('--stddev',
                        type=float,
                        nargs='+',
                        help='stddev value',
                        default=None)
    parser.add_argument('--divisor',
                        type=float,
                        help='divisor value',
                        default=None)
    parser.add_argument('--width',
                        type=int,
                        help=' the width of resize size ',
                        default=None)
    parser.add_argument('--height',
                        type=int,
                        help=' the height of resize size ',
                        default=None)
    parser.add_argument('--color_format',
                        type=str,
                        help='color_format',
                        default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    image_path = args.image_path
    npy_path = args.save_path
    image_to_npy = ImageToNpy(image_path, npy_path)
    config = {
        'width': args.width,
        'height': args.height,
        'mean': args.mean,
        'stddev': args.stddev,
        'divisor': args.divisor,
        'color_format': args.color_format,
        }
    for key, value in config.items():
        if value is not None:
            image_to_npy.set_config(key, value)

    image_to_npy.image2npy()
    print("save to ", os.path.abspath(npy_path))

    # data = np.load(npy_path)
    # print(data.shape)
    # for i in range (data.shape[0]):
    #     for j in range (20):
    #         print(i, data[i, j, 0])


if __name__ == '__main__':
    main()
    