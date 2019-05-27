from __future__ import print_function

import os
from os.path import join

import numpy as np
import torch.utils.data as data
from PIL import Image

from .utils import download_url, check_integrity, list_dir, list_files


class Omniglot(data.Dataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """
    folder = 'omniglot-py'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    def __init__(self, root, background=True,
                 transform=None, target_transform=None,
                 download=False, train=True, all=False):
        self.root = join(os.path.expanduser(root), self.folder)
        self.background = background
        self.transform = transform
        self.target_transform = target_transform
        self.images_cached = {}

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                for a in self._alphabets], [])
        self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
                                  for idx, character in enumerate(self._characters)]
        self._flat_character_images = sum(self._character_images, [])
        self.data = [x[0] for x in self._flat_character_images]
        self.targets = [x[1] for x in self._flat_character_images]
        self.data2 = []
        self.targets2 = []
        self.new_flat = []
        for a in range(int(len(self.targets) / 20)):
            start = a * 20
            if train:
                for b in range(start, start + 15):
                    self.data2.append(self.data[b])
                    self.targets2.append(self.targets[b])
                    self.new_flat.append(self._flat_character_images[b])
                    # print(self.targets[start+b])
            else:
                for b in range(start + 15, start + 20):
                    self.data2.append(self.data[b])
                    self.targets2.append(self.targets[b])
                    self.new_flat.append(self._flat_character_images[b])

        if all:
            pass
        else:
            self._flat_character_images = self.new_flat
            self.targets = self.targets2
            print(self.targets[0:30])
            self.data = self.data2

        print("Total classes = ", np.max(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        # image_name, character_class = self._flat_character_images[index]
        image_name = self.data[index]
        character_class = self.targets[index]
        image_path = join(self.target_folder, self._characters[character_class], image_name)
        if image_path not in self.images_cached:

            image = Image.open(image_path, mode='r').convert('L')
            if self.transform:
                image = self.transform(image)

            self.images_cached[image_path] = image
        else:
            image = self.images_cached[image_path]

        # if self.transform:
        #     image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def _cache_data(self):
        pass

    def _check_integrity(self):
        zip_filename = self._get_target_folder()
        if not check_integrity(join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
            return False
        return True

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        filename = self._get_target_folder()
        zip_filename = filename + '.zip'
        url = self.download_url_prefix + '/' + zip_filename
        download_url(url, self.root, zip_filename, self.zips_md5[filename])
        print('Extracting downloaded file: ' + join(self.root, zip_filename))
        with zipfile.ZipFile(join(self.root, zip_filename), 'r') as zip_file:
            zip_file.extractall(self.root)

    def _get_target_folder(self):
        return 'images_background' if self.background else 'images_evaluation'
