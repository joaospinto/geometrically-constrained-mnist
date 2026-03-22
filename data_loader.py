import os
import urllib.request
import gzip
import numpy as np

# URLs for the MNIST dataset
URL_BASE = 'http://yann.lecun.com/exdb/mnist/'
FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz'
}

def download_mnist(data_dir='./data'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    for filename in FILES.values():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            # Use alternative mirror if LeCun's site is slow/down (often happens)
            # Try main site first.
            try:
                urllib.request.urlretrieve(URL_BASE + filename, filepath)
            except Exception:
                # Backup mirror (often used in tutorials)
                backup_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
                print(f"Primary failed. Trying backup mirror for {filename}...")
                urllib.request.urlretrieve(backup_url + filename, filepath)

def load_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        # Magic number, number of images, rows, cols
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=np.dtype('>i4'), count=4)
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
        return images.astype(np.float32) / 255.0

def load_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        # Magic number, number of items
        magic, num = np.frombuffer(f.read(8), dtype=np.dtype('>i4'), count=2)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def get_mnist_data(data_dir='./data'):
    download_mnist(data_dir)
    
    train_images = load_images(os.path.join(data_dir, FILES['train_images']))
    train_labels = load_labels(os.path.join(data_dir, FILES['train_labels']))
    test_images = load_images(os.path.join(data_dir, FILES['test_images']))
    test_labels = load_labels(os.path.join(data_dir, FILES['test_labels']))
    
    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    tr_i, tr_l, te_i, te_l = get_mnist_data()
    print(f"Train images: {tr_i.shape}, Train labels: {tr_l.shape}")
    print(f"Test images: {te_i.shape}, Test labels: {te_l.shape}")
