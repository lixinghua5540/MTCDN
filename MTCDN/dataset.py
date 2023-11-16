from os import listdir
from os.path import join
import random
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "Image")
        self.b_path = join(image_dir, "Image2")
        self.label_path = join(image_dir, "label")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        label = Image.open(join(self.label_path, self.image_filenames[index]))
        a = np.array(a)/255.0
        b = np.array(b)/255.0
        a = np.transpose(a, (2, 0, 1))
        b = np.transpose(b, (2, 0, 1))
        label = np.array(label)
        #label = label[:, :, 0]
        lbl = np.where(label > 0, 1, label)
        # #
        # a = a.resize((286, 286), Image.BICUBIC)
        # b = b.resize((286, 286), Image.BICUBIC)
        # a = transforms.ToTensor()(a)
        # b = transforms.ToTensor()(b)
        # w_offset = random.randint(0, max(0, 286 - 256 - 1))
        # h_offset = random.randint(0, max(0, 286 - 256 - 1))
        #
        # a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        # b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        #
        # a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        # b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        # if random.random() < 0.5:
        #     idx = [i for i in range(a.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     a = a.index_select(2, idx)
        #     b = b.index_select(2, idx)

        if self.direction == "a2b":#direction 在这里体现区别
            return a, b,lbl
        else:
            return b, a,lbl

    def __len__(self):
        return len(self.image_filenames)
