import os
import cv2
import numpy as np

def image2npy(image_path, npy_path):
    means = [0.4914, 0.4822, 0.4465]
    stddev = [0.247, 0.243, 0.261]
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (np.array(image, np.float32) / 255 - means) / stddev
    image = image.transpose([2, 0, 1])
    image = image.astype(np.float32)
    print(image)
    np.save(npy_path, image)

def main():
    image_path = r"data.png"
    npy_path = r"data.npy"
    image2npy(image_path, npy_path)
    print("save to ", os.path.abspath(npy_path))

    # data = np.load(npy_path)
    # print(data.shape)
    # for i in range (data.shape[0]):
    #     for j in range (20):
    #         print(i, data[i, j, 0])

if __name__ == '__main__':
    main()
    