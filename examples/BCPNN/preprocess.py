import numpy as np
import os
import struct
import gzip
from urllib.request import urlretrieve
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlretrieve
import gzip
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle

class MNISTDataset:
    """ MNIST loader for train/test images and labels. """

    def __init__(
        self,
        root="Data/mnist",
        url="https://github.com/fgnt/mnist/raw/refs/heads/master",
        download=True,
        prepare=True,
        flatten=True,
        one_hot_labels=True,
    ):
        self.root = Path(root)
        self.url = url.rstrip("/")
        self.download_dir = self.root / "Downloads"
        self.flatten = flatten
        self.one_hot_labels = one_hot_labels

        self.raw_files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        }
        
        self.bin_files = {
            "train_images": self.root / "mnist_train_images.bin",
            "train_labels": self.root / "mnist_train_labels.bin",
            "test_images": self.root / "mnist_test_images.bin",
            "test_labels": self.root / "mnist_test_labels.bin",            
        }

        self.root.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        if download:
            self.download_if_needed()

        self.train_images = self._read_images(self.download_dir / self.raw_files["train_images"], 60000)
        self.test_images = self._read_images(self.download_dir / self.raw_files["test_images"], 10000)

        self.train_labels = self._read_labels(self.download_dir / self.raw_files["train_labels"], 60000)
        self.test_labels = self._read_labels(self.download_dir / self.raw_files["test_labels"], 10000)

        if self.flatten:
            self.train_images = self.train_images.reshape(60000, -1)
            self.test_images = self.test_images.reshape(10000, -1)

        if self.one_hot_labels:
            self.train_labels = np.eye(10, dtype=np.float32)[self.train_labels]
            self.test_labels = np.eye(10, dtype=np.float32)[self.test_labels]

    def download_file(self, url, out_path):
        """Download file if it does not already exist."""
        out_path = Path(out_path)
        if out_path.exists():
            return
        print(f"Downloading {url} -> {out_path}")
        urlretrieve(url, out_path)

    def download_if_needed(self):
        """Download all raw MNIST gzip files."""
        for fname in self.raw_files.values():
            self.download_file(f"{self.url}/{fname}", self.download_dir / fname)

    def write_to_file(self, arr, out_path, dtype):
        if Path(out_path).exists():
            return
        arr.astype(dtype).tofile(out_path)

    def save_binary_files(self):
        """Save train/test images and labels as binary files."""
        self.write_to_file(self.train_images, self.bin_files["train_images"], dtype=np.float32)
        self.write_to_file(self.train_labels, self.bin_files["train_labels"], dtype=np.float32)
        self.write_to_file(self.test_images, self.bin_files["test_images"], dtype=np.float32)
        self.write_to_file(self.test_labels,  self.bin_files["test_labels"], dtype=np.float32)

    def get_image(self, idx, split="train"):
        """Return one image from the requested split."""
        if split == "train":
            return self.train_images[idx]
        elif split == "test":
            return self.test_images[idx]
        else:
            print (f"Unknown split {split}!")

    def get_label(self, idx, split="train", one_hot=None):
        """Return one label from the requested split."""
        if split == "train":
            y = self.train_labels[idx]
        elif split == "test":
            y = self.test_labels[idx]
        else:
            print (f"Unknown split {split}!")

        if one_hot is None:
            return y

        if one_hot and np.ndim(y) == 0:
            return np.eye(10, dtype=np.float32)[int(y)]

        if (not one_hot) and np.ndim(y) > 0:
            return int(np.argmax(y))

        return y

    @staticmethod
    def _read_images(path, n):
        """Read raw MNIST images and normalize to [0, 1]."""
        with gzip.open(path, "rb") as f:
            f.read(16)
            buf = f.read(n * 28 * 28)
        return np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(n, 28, 28) / 255.0

    @staticmethod
    def _read_labels(path, n):
        """Read raw MNIST labels as integer class ids."""
        with gzip.open(path, "rb") as f:
            f.read(8)
            buf = f.read(n)
        return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    
class PixelwiseGMM:
    
    def __init__(self, 
                 n_components=10, 
                 covariance_type="diag", 
                 random_state=0, 
                 verbose=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.verbose = verbose
        self.model = GaussianMixture(n_components=n_components, 
                                     covariance_type=covariance_type, 
                                     tol=1e-3, 
                                     n_init=10, 
                                     verbose=verbose)
        
    def fit(self, images, num_pixel=100000):
        """
        Fit GMM on RGB pixel values.

        Parameters
        ----------
        images : np.ndarray
            Shape could be (N, H*W, 3) or (N, H*W)
        """
        print ("PerpixelGMM", images.shape)
        # Flatten data (squash all pixels)
        if images.shape[-1] == 3:
            in_channels = 3
        else:
            in_channels = 1
        images_flatten = images.reshape(-1, in_channels)
        # Shuffle data and select subset for fitting
        randidx = np.arange(images_flatten.shape[0])
        np.random.shuffle(randidx)
        # Fit model
        self.model.fit(images_flatten[randidx][:num_pixel])
        # Check fit
        if self.verbose > 0:
            print ("PerpixelGMM cluster centers\n")
            print ("Responsibility", np.round(self.model.weights_, in_channels))
            print ("Means", np.round(self.model.means_, in_channels))
            print ("Covariance", np.round(self.model.covariances_, in_channels))

    def get_sample(self, image, split="train"):
        """Return one image from the requested split."""
        if image.shape[-1] == 3:
            in_channels = 3
        else:
            in_channels = 1
        image = image.reshape(-1, in_channels)
        postr = self.model.predict_proba(image)
        return postr.flatten()
    
    def save_model(self, filename):
        with open(filename,'wb') as f:
            pickle.dump(self.model,f)
            
    def load_model(self, filename):
        with open(filename,'rb') as f:
            self.model = pickle.load(f)
