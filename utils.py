import datetime
import os
from PIL import Image, ImageStat
from concurrent.futures import ThreadPoolExecutor


def format_time(elapsed):
    """
    :param elapsed: time in seconds
    :return: formatted time string in hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def remove_grayscale_images(dataset_folder, dataset_type):
    """
    :param dataset_folder: path to the dataset folder
    :param dataset_type: train, val or test split
    """
    # parameters value check dataset_type: map train, val, test to train2017, val2017, test2017
    if dataset_type == 'train':
        dataset_type = 'train2017'
    elif dataset_type == 'val':
        dataset_type = 'val2017'
    elif dataset_type == 'test':
        dataset_type = 'test2017'
    else:
        raise Exception(
            'Invalid dataset type: ' + dataset_type + '. Must be train, val, or test.')

    path = os.path.join(dataset_folder, dataset_type)

    # Initialize a counter for the number of grayscale images
    gray_count = 0

    def remove_gray_image(image_path):
        # load image
        img = Image.open(image_path).convert('RGB')
        # check if img is grayscale remove it and update counter
        stat = ImageStat.Stat(img)
        if sum(stat.sum) / 3 == stat.sum[0]:
            img.close()
            os.remove(image_path)
            return 1
        return 0

    # Iterate through the path directory and remove the grayscale images
    executor = ThreadPoolExecutor()
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            future = executor.submit(remove_gray_image, os.path.join(path, file))
            gray_count += future.result()

    executor.shutdown(wait=True)
    print('Removed ' + str(gray_count) + ' grayscale images from ' + path)


if __name__ == '__main__':
    dataset_folder = 'C:\\Users\\loren\\Datasets\\coco2017'
    dataset_type = 'train'
    remove_grayscale_images(dataset_folder, dataset_type)
