import numpy as np
from dezero.models import VGG16
if __name__ == '__main__':
    model = VGG16(pretrained=True)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)  # 虚拟数据
    model.plot(x, to_file='VGG16.png', from_file='VGG16.dot')