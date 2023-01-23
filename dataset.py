# create Dataset class to load coco dataset
import os.path
from torch.utils.data import Dataset
from PIL import Image
from skimage.color import rgb2lab
from torchvision.transforms import ToTensor, Compose


class CocoDataset(Dataset):

    def __init__(self, dataset_folder, dataset_type, transforms=None, use_lab_colorspace=False,
                 frac=1.0):
        """
        :param dataset_folder: path to the dataset folder
        :param dataset_type: train, val or test split
        :param transforms: transforms to apply to the input image
        :param use_lab_colorspace: whether to use LAB colorspace
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
            raise Exception(
                'Invalid dataset type: ' + dataset_type + '. Must be train, val, or test.')

        self.path = os.path.join(dataset_folder, dataset_type)
        self.transforms = transforms
        self.use_lab_colorspace = use_lab_colorspace
        self.frac = frac
        self.to_tensor = ToTensor()

        # remove ToTensor() from the transform list if present
        if self.transforms is not None:
            tr_list = [tr for tr in self.transforms.transforms if not isinstance(tr, ToTensor)]
            self.transforms = Compose(tr_list) if len(tr_list) > 0 else None

        self.images = []

        # Iterate through the path directory and add the paths for all the images in the directory
        for file in os.listdir(self.path):
            if file.endswith('.jpg'):
                self.images.append(os.path.join(self.path, file))

    def __len__(self):
        return int(len(self.images) * self.frac)

    def __getitem__(self, idx):
        # load image
        img = Image.open(self.images[idx])
        # check if img is grayscale (there are a couple of hundred grayscale images in the dataset)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # apply transforms, still a PIL image
        if self.transforms is not None:
            img = self.transforms(img)

        # convert to LAB colorspace and extract L and AB channels
        if self.use_lab_colorspace:
            img = rgb2lab(img).astype("float32")
            img = self.to_tensor(img)  # not scaled since dtype is float32
            L = img[[0], :, :] / 50. - 1.  # scale to [-1, 1]
            ab = img[[1, 2], :, :] / 110.  # scale to [-1, 1]
            return L, ab

        # get grayscale and rgb images
        else:
            gray, color = img.convert('L'), img
            # convert to tensors in range [0, 1]
            gray = self.to_tensor(gray)
            color = self.to_tensor(color)
            return gray, color


if __name__ == '__main__':
    # test dataset
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # compose resize and to tensor transforms
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

    dataset = CocoDataset(dataset_folder='C:\\Users\\loren\\Datasets\\coco2017', dataset_type='val',
                          use_lab_colorspace=False, transforms=transforms)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_x, batch_y in dataloader:
        print(batch_x.shape)
        print(batch_x[0])
        print(batch_y.shape)
        print(batch_y[0])
        break
