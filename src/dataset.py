import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
# from sklearn import preprocessing

root = os.getcwd()
colab_root = '/content/drive/My Drive/PPDAE'
exalearn_root = '/home/jorgemarpa/data/PPD'


class MyRotationTransform:
    """Rotate by a random N times 90 deg."""

    def __init__(self):
        pass

    def __call__(self, x):
        shape = x.shape
        return np.rot90(x, np.random.choice([0, 1, 2, 3]),
                        axes=[-2, -1]).copy()


class MyFlipVerticalTransform:
    """Random vertical flip."""

    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, x):
        if np.random.uniform() >= self.prob:
            return np.flip(x, -2).copy()
        else:
            return x


class MyNormTransform:
    """Normalization."""

    def __init__(self, mean=0., std=1.):
        self.mean = np.array(mean)
        self.std = np.array(std)
        if self.mean.ndim == 1:
            self.mean = self.mean[None, :, None, None]
        if self.std.ndim == 1:
            self.std = self.std[None, :, None, None]

    def __call__(self, x):
        return (x - self.mean) / self.std


# load pkl synthetic light-curve files to numpy array
class ProtoPlanetaryDisks(Dataset):
    def __init__(self, machine='local', transform=True,
                 subsample=False, img_norm=True):
        if machine == 'local':
            ppd_path = '%s/data/PPD' % (root)
        elif machine == 'colab':
            ppd_path = '%s/' % (colab_root)
        elif machine == 'exalearn':
            ppd_path = '%s/' % (exalearn_root)
        else:
            raise('Wrong host, please select local, colab or exalearn')

        self.meta = np.load('%s/param_arr.npy' % (ppd_path))
        self.meta = self.meta.astype(np.float32)
        self.meta_names = ['m_dust', 'Rc', 'f_exp', 'H0',
                           'Rin', 'sd_exp', 'a_max', 'inc']
        self.img_norm = img_norm
        if not self.img_norm:
            self.imgs = np.load('%s/img_array.npy' % (ppd_path))
            self.imgs = np.expand_dims(self.imgs, axis=1)
        else:
            self.imgs = np.load('%s/img_norm_array.npy' % (ppd_path))
        self.imgs = self.imgs.astype(np.float32)
        # imgs has shape [batch_size, channels, height, width]
        if subsample:
            idx = np.random.randint(0, self.meta.shape[0], 1000)
            self.imgs = self.imgs[idx]
            self.meta = self.meta[idx]
        self.img_dim = self.imgs[0].shape[-1]
        self.transform = transform
        self.transform_fx = torchvision.transforms.Compose([
            MyRotationTransform(),
            MyFlipVerticalTransform()])


    def __getitem__(self, index):
        imgs = self.imgs[index]
        meta = self.meta[index]
        if self.transform:
            imgs = self.transform_fx(imgs)
        return imgs, meta

    def __len__(self):
        return len(self.imgs)

    def get_dataloader(self, batch_size=32, shuffle=True,
                       test_split=0.2, random_seed=42):

        np.random.seed(random_seed)
        if test_split == 0.:
            train_loader = DataLoader(self, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)
            test_loader = None
        else:
            # Creating data indices for training and test splits:
            dataset_size = len(self)
            indices = list(range(dataset_size))
            split = int(np.floor(test_split * dataset_size))
            if shuffle:
                np.random.shuffle(indices)
            train_indices, test_indices = indices[split:], indices[:split]
            del indices, split

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            train_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=train_sampler,
                                      drop_last=False)
            test_loader = DataLoader(self, batch_size=batch_size,
                                     sampler=test_sampler,
                                     drop_last=False)

        return train_loader, test_loader


class MNIST(Dataset):
    def __init__(self, machine='local'):
        if machine == 'local':
            mnist_path = '%s/data/' % (root)
        elif machine == 'colab':
            mnist_path = '%s/' % (colab_root)
        elif machine == 'exalearn':
            mnist_path = '%s/' % (exalearn_root)
        self.train = torchvision.datasets.MNIST(
            mnist_path, train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.test = torchvision.datasets.MNIST(
            mnist_path, train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.img_dim = self.test[0][0].shape[-1]

    def __getitem__(self, index):
        return self.test[index]

    def __len__(self):
        return len(self.train) + len(self.test)

    def get_dataloader(self, batch_size=32, shuffle=True,
                       test_split=.2, random_seed=32):

        train_loader = DataLoader(self.train,
                                  batch_size=batch_size,
                                  shuffle=shuffle)

        test_loader = DataLoader(self.test,
                                 batch_size=batch_size,
                                 shuffle=shuffle)

        return train_loader, test_loader
