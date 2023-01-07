# create Dataset class to load coco dataset
import os.path
from torch.utils.data import Dataset
from PIL import Image


class CocoDataset(Dataset):

    def __init__(self, dataset_folder, dataset_type, transform_x=None, transform_y=None, frac=1.0):
        """
        :param dataset_folder: path to the dataset folder
        :param dataset_type: train, val or test split
        :param transform_x: transform to apply to the input image
        :param transform_y: transform to apply to the target image
        :param frac: fraction of the dataset to use
        """
        # parameters value check dataset_type: map train, val, test to train2017, val2017, test2017
        if dataset_type == 'train':
            dataset_type = 'train2017'
        elif dataset_type == 'val':
            dataset_type = 'val2017'
        elif dataset_type == 'test':
            dataset_type = 'test2017'
        else:
            raise Exception('Invalid dataset type: ' + dataset_type + '. Must be train, val, or test.')

        self.path = os.path.join(dataset_folder, dataset_type)
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.frac = frac

        self.images = []

        # Iterate through the path directory and add the paths for all the images in the directory
        for file in os.listdir(self.path):
            if file.endswith('.jpg'):
                self.images.append(os.path.join(self.path, file))

    def __len__(self):
        return int(len(self.images)*self.frac)

    def __getitem__(self, idx):
        # load image
        img = Image.open(self.images[idx])
        # check if img is grayscale (there are a couple of hundred grayscale images in the dataset)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        x, y = img, img

        # apply transforms
        if self.transform_x is not None:
            x = self.transform_x(img)

        if self.transform_y is not None:
            y = self.transform_y(img)

        return x, y


if __name__ == '__main__':
    # test dataset
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # compose resize and to tensor transforms
    transform_x = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(),
                                      transforms.ToTensor()])
    transform_y = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    dataset = CocoDataset(dataset_folder='C:\\Users\\loren\\Datasets\\coco2017', dataset_type='val',
                          transform_x=transform_x, transform_y=transform_y)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_x, batch_y in dataloader:
        print(batch_x.shape)
        print(batch_x[0])
        print(batch_y.shape)
        print(batch_y[0])
        break
