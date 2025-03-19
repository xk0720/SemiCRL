import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

def get_dataset_path(dataset='pancreas'):
    files = ['train.txt', 'test.txt']
    return ['/'.join(['data_lists', dataset, f]) for f in files]


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample[0], sample[1]
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return [image, label]
        # return {'image': image, 'label': label}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]


class Pancreas(Dataset):
    """ Pancreas Dataset """

    def __init__(self, base_dir):
        self._base_dir = base_dir
        # self.split = split

        tr_transform = Compose([
            # RandomRotFlip(),
            RandomCrop((96, 96, 96)),
            # RandomNoise(),
            ToTensor()
        ])

        # if no_crop:
        #     test_transform = Compose([
        #         # CenterCrop((160, 160, 128)),
        #         CenterCrop((96, 96, 96)),
        #         ToTensor()
        #     ])
        # else:
        #     test_transform = Compose([
        #         CenterCrop((96, 96, 96)),
        #         ToTensor()
        #     ])

        data_list_paths = get_dataset_path()

        # if split == 'train_lab':
        #     data_path = data_list_paths[0]
        #     self.transform = tr_transform
        # elif split == 'train_unlab':
        #     data_path = data_list_paths[1]
        #     self.transform = test_transform  # tr_transform
        # else:
        #     data_path = data_list_paths[2]
        #     self.transform = test_transform

        data_path = data_list_paths[0]
        self.transform = tr_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [self._base_dir + "/{}".format(item.strip()) for item in self.image_list]
        # print("Split : {}, total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        # if self.split == 'train_lab':
        #     return len(self.image_list) * 5
        # else:
        #     return len(self.image_list)
        return len(self.image_list)

    def __getitem__(self, idx):
        # image_path = self.image_list[idx % len(self.image_list)]
        image_path = self.image_list[idx]
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
        samples = image, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_, label_ = tr_samples
        return image_.float(), label_.long(), idx


# class Pancreas(Dataset):
#     def __init__(self, root_dir, transform=None, normalize=False):
#         super(Dataset, self).__init__()
#         self.img_path = os.path.join(root_dir, 'Images')
#         self.label_path = os.path.join(root_dir, 'Labels')
#
#         self.image_list = sorted(os.listdir(self.img_path))  # list
#
#         self.transform = transform
#         self.normalize = normalize
#         self.min_value = -125
#         self.max_value = 275
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def __getitem__(self, index):
#         img_file = os.path.join(self.img_path, self.image_list[index])
#         label_file = self.label_path + '/label' + self.image_list[index]
#
#         # image = np.load(img_file)
#         # h5f = h5py.File(img_file, 'r')
#         # image = h5f['image'][:]
#         # image = image.astype('float')
#
#         image = nib.load(img_file)
#         image = image.get_fdata() # load the numpy array
#         # apply the window
#         image[image < self.min_value] = self.min_value
#         image[image > self.max_value] = self.max_value
#         # normalize the image
#         image = (image - self.min_value) / (self.max_value - self.min_value) # range [0, 1]
#
#         label = nib.load(label_file)
#         label = label.get_fdata()
#         # h5f = h5py.File(label_file, 'r')
#         # label = h5f['label'][:]
#
#         sample = {'image': image, 'label': label}
#         if self.transform:
#             sample = self.transform(sample)
#
#         if self.normalize:
#             mean = torch.mean(sample['image'])
#             var = torch.var(sample['image'])
#             sample['image'] = (sample['image'] - mean) / var
#
#         return sample


# def get_loader(args):
#     train_transform = transforms.Compose(
#         [
#             transforms.LoadImaged(keys=["image", "label"]),
#             transforms.AddChanneld(keys=["image", "label"]),
#             transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
#             # transforms.Spacingd(
#             #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
#             # ),
#             # transforms.ScaleIntensityRanged(
#             #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
#             # ),
#             transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
#             # Based on 'pos=1, neg=1', the probability of selecting the foreground pixel as the cropping center is 0.5.
#             transforms.RandCropByPosNegLabeld(
#                 keys=["image", "label"],
#                 label_key="label",
#                 spatial_size=(args.roi_x, args.roi_y, args.roi_z),
#                 pos=1,
#                 neg=0,
#                 num_samples=1,
#                 # set 'num_samples' to 1, meaning generate one cropped result for one image, of course we can set 2 or more.
#                 image_key="image",
#                 image_threshold=0,
#             ),
#             transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
#             transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
#             transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
#             transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
#             # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
#             # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
#             transforms.ToTensord(keys=["image", "label"]),
#             # transforms.NormalizeIntensity(subtrahend=0.5, divisor=0.5),
#         ]
#     )
#
#     val_transform = transforms.Compose(
#         [
#             transforms.LoadImaged(keys=["image", "label"]),
#             transforms.AddChanneld(keys=["image", "label"]),
#             transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
#             transforms.Spacingd(
#                 keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
#             ),
#             transforms.ScaleIntensityRanged(
#                 keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
#             ),
#             transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
#             transforms.ToTensord(keys=["image", "label"]),
#         ]
#     )
#
#     datalist_json = os.path.join(args.data_dir, args.json_list)  # json_list == 'dataset_0.json'
#     datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=args.data_dir)
#
#     if args.use_normal_dataset == 1:
#         train_ds = data.Dataset(data=datalist, transform=train_transform)
#     else:
#         train_ds = data.CacheDataset(
#             data=datalist, transform=train_transform, cache_num=len(datalist), cache_rate=1.0, num_workers=args.workers
#         )
#
#     # randomly split the unlabeled/labeled data
#     # unlabeled_indices, labeled_indices = train_test_split(
#     #     np.linspace(0, len(datalist) - 1, len(datalist)).astype('int'),
#     #     test_size=int(len(datalist) * args.semi_ratio),
#     #     random_state=42,
#     # )
#
#     # create batch sampler (labeled, unlabeled)
#     labeled_indices = list(range(int(len(datalist) * args.semi_ratio)))  # 12
#     unlabeled_indices = list(range(int(len(datalist) * args.semi_ratio), len(datalist)))  # 48
#
#     train_sampler = TwoStreamBatchSampler(
#         primary_indices=labeled_indices,
#         secondary_indices=unlabeled_indices,
#         batch_size=args.batch_size,
#         secondary_batch_size=args.batch_size - args.labeled_bs,
#     )
#
#     train_loader = data.DataLoader(
#         train_ds,
#         # batch_size=args.batch_size,
#         # shuffle=(train_sampler is None),
#         # num_workers=args.workers,
#         batch_sampler=train_sampler,
#         pin_memory=True,
#         # persistent_workers=True,
#     )
#     val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=args.data_dir)
#     val_ds = data.Dataset(data=val_files, transform=val_transform)
#
#     val_loader = data.DataLoader(
#         val_ds,
#         batch_size=1,
#         shuffle=False,
#         # num_workers=args.workers,
#         # sampler=val_sampler,
#         pin_memory=True,
#         # persistent_workers=True,
#     )
#
#     return [train_loader, val_loader]

