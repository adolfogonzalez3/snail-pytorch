'''A module that contains a class for loading the omniglot data set.'''
import shutil
from pathlib import Path
from urllib.request import urlopen


import torch.utils.data as data
import numpy as np
import errno
import os
from PIL import Image
import torch

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

IMG_CACHE = {}


class OmniglotDataset(data.Dataset):
    vinalys_baseurl = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/'
    vinyals_split_sizes = {
        'test': vinalys_baseurl + 'test.txt',
        'train': vinalys_baseurl + 'train.txt',
        'trainval': vinalys_baseurl + 'trainval.txt',
        'val': vinalys_baseurl + 'val.txt',
    }

    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    splits_folder = Path('splits', 'vinyals')
    raw_folder = 'raw'
    processed_folder = 'data'

    def __init__(self, mode='train', root='../dataset/omniglot', transform=None, target_transform=None, download=True):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(OmniglotDataset, self).__init__()
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it'
            )

        self.classes = get_current_classes(
            self.root / self.splits_folder / (mode + '.txt')
        )
        self.all_items = find_items(
            self.root / self.processed_folder, self.classes
        )
        self.idx_classes = index_classes(self.all_items)

        paths, self.y = zip(*[self.get_path_label(pl)
                              for pl in range(len(self))])

        self.x = [load_img(path, i) for i, path in enumerate(paths)]

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        filename = self.all_items[index][0]
        rot = self.all_items[index][-1]
        img = str.join('/', [self.all_items[index][2], filename]) + rot
        target = self.idx_classes[self.all_items[index]
                                  [1] + self.all_items[index][-1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        return (self.root / self.processed_folder).exists()

    def download(self):
        '''Download Omniglot if not exists.'''
        if self._check_exists():
            return

        (self.root / self.splits_folder).mkdir(parents=True, exist_ok=True)
        (self.root / self.raw_folder).mkdir(exist_ok=True)
        (self.root / self.processed_folder).mkdir(exist_ok=True)

        for k, url in self.vinyals_split_sizes.items():
            print('== Downloading ' + url)
            filename = url.rpartition('/')[-1]
            download(url, self.root / self.splits_folder / filename)

        orig_root = self.root / self.raw_folder
        for url in self.urls:
            print('== Downloading ' + url)
            filename = url.rpartition('/')[2]
            file_path = self.root / self.raw_folder / filename
            download(url, file_path)
            print(f'== Unzip from {file_path} to {orig_root}')
            shutil.unpack_archive(file_path, orig_root)
        file_processed = str(self.root / self.processed_folder)
        for image_type in ['images_background', 'images_evaluation']:
            for path in (orig_root / image_type).iterdir():
                shutil.move(str(path), file_processed)
            os.rmdir(orig_root / image_type)
        print("Download finished.")


def find_items(root_dir, classes):
    root_dir = Path(root_dir)
    retour = []
    rots = ['/rot000', '/rot090', '/rot180', '/rot270']

    for path in root_dir.rglob('*.png'):
        root = str(path.parent)
        label = f'{path.parts[-3]}/{path.parts[-2]}'
        for rot in rots:
            if label + rot in classes:
                retour.append((path.name, label, root, rot))
    print("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] + i[-1] in idx):
            idx[i[1] + i[-1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx


def get_current_classes(fname):
    with open(fname) as f:
        classes = f.read().splitlines()
    return classes


def load_img(path, idx):
    path, rot = path.split('/rot')
    if path in IMG_CACHE:
        x = IMG_CACHE[path]
    else:
        x = Image.open(path)
        IMG_CACHE[path] = x
    x = x.rotate(float(rot))
    x = x.resize((28, 28))

    shape = 1, x.size[0], x.size[1]
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - torch.from_numpy(x)
    x = x.transpose(0, 1).contiguous().view(shape)

    return x

def download(url, dest):
    '''Download a file and write to dest.'''
    with urlopen(url) as response, open(dest, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
