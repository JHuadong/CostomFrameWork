import numpy as np
from PIL import Image
import dezero
from dezero.models import VGG16
import pathlib

if __name__ == '__main__':
    url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
    img_path = dezero.utils.get_file(url,
                                     cache_dir=pathlib.Path(__file__).parent.parent.absolute() / 'pictures',
                                     file_name='zebra.jpg')

    img = Image.open(img_path)
    #img.show()

    x = VGG16.preprocess(img)
    x = x[np.newaxis]  # Add an axis for small-batch processing
    model = VGG16(pretrained=True)
    with dezero.test_mode():
        y = model(x)
    predict_id = np.argmax(y.data)
    model.plot(x, from_file='vgg.dot', to_file='vgg.pdf')  # Visualisation of Computational Graphs
    labels = dezero.datasets.ImageNet.labels()  # the labels from ImageNet
    print(labels[predict_id])