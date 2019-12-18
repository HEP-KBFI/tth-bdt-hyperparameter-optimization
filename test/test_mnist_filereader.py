'''Testing the MNIST filereader functions'''
import os
import urllib
import gzip
import shutil
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader as mf

dir_path = os.path.dirname(os.path.realpath(__file__))
resourcesDir = os.path.join(dir_path, 'resources')
tmp_folder = os.path.join(resourcesDir, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

main_url = 'http://yann.lecun.com/exdb/mnist/'
train_images = 'train-images-idx3-ubyte'
train_labels = 'train-labels-idx1-ubyte'
test_images = 't10k-images-idx3-ubyte'
test_labels = 't10k-labels-idx1-ubyte'

file_list = [train_labels, train_images, test_labels, test_images]
sample_dir = os.path.join(tmp_folder, 'samples_mnist')
nthread = 2
os.makedirs(sample_dir)

for file in file_list:
    file_loc = os.path.join(sample_dir, file)
    file_url = os.path.join(main_url, file + '.gz')
    urllib.urlretrieve(file_url, file_loc + '.gz')
    with gzip.open(file_loc + '.gz', 'rb') as f_in:
        with open(file_loc, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def test_read_images():
    '''Testing read_images function'''
    image_file = os.path.join(sample_dir, 't10k-images-idx3-ubyte')
    result = mf.read_images(image_file)
    assert len(result) == 10000


def test_read_labels():
    '''Testing read_labels function'''
    label_file = os.path.join(sample_dir, 't10k-labels-idx1-ubyte')
    result = mf.read_labels(label_file)
    assert len(result) == 10000


def test_read_dataset():
    '''Testing read_dataset function'''
    image_file = os.path.join(sample_dir, 't10k-images-idx3-ubyte')
    label_file = os.path.join(sample_dir, 't10k-labels-idx1-ubyte')
    result = mf.read_dataset(image_file, label_file)
    assert len(result) == 2
    for element in result:
        assert len(element) == 10000


def test_create_datasets():
    '''Testing create_datasets function'''
    # INCOMPLETE
    result = mf.create_datasets(sample_dir, 16)
    assert len(result['training_labels']) == 60000
    assert len(result['testing_labels']) == 10000
