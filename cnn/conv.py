import numpy as np


class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.sz = 3
        self.filters = np.random.randn(num_filters, \
                                       self.sz, self.sz)/(self.sz**2)

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h-(self.sz-1)):
            for j in range(w-(self.sz-1)):
                im_region = image[i:(i + self.sz), j:(j + self.sz)]
                yield im_region, i, j

    def forward(self, input):
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))


c = Conv3x3(1)
matrix = np.arange(27).reshape(3,3,3)
print(matrix)
print("="*10)
print(np.sum(matrix, axis=(1)))
print(np.sum(matrix, axis=(1,2)))
