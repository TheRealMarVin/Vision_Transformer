class ToPatch:
    def __init__(self, img_size, patch_size):
        self.img_size = img_size
        self.patch_size = patch_size

        # determine patch count
        self.patch_count = int((self.img_size[1] * self.img_size[2]) / (self.patch_size[0] * self.patch_size[1]))

    def __call__(self, x):

        # unfold channels
        x = x.data.unfold(dimension=0, size=self.img_size[0], step=self.img_size[0])
        # unfold width
        x = x.data.unfold(dimension=1, size=self.patch_size[0], step=self.patch_size[0])
        # unfold height
        x = x.data.unfold(dimension=2, size=self.patch_size[1], step=self.patch_size[1])

        x = x.reshape(self.patch_count, self.img_size[0], self.patch_size[0], self.patch_size[1])

        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'
