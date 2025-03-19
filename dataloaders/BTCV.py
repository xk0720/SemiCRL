import os
import numpy as np
import torch
from monai import data, transforms
from monai.data import load_decathlon_datalist
from sklearn.model_selection import train_test_split
from dataloaders.utils import TwoStreamBatchSampler


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd( #
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            # Based on 'pos=1, neg=1', the probability of selecting the foreground pixel as the cropping center is 0.5.
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=1,
                # set 'num_samples' to 1, meaning generate one cropped result for one image, of course we can set 2 or more.
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)

    if args.use_normal_dataset == 1:
        train_ds = data.Dataset(data=datalist, transform=train_transform)
    else:
        train_ds = data.CacheDataset(
            data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
        )

    # randomly split the unlabeled/labeled data
    unlabeled_indices, labeled_indices = train_test_split(
        np.linspace(0, len(datalist) - 1, len(datalist)).astype('int'),
        test_size=int(len(datalist) * args.semi_ratio),
        random_state=42,
    )

    # train_sampler = Sampler(train_ds) if args.distributed else None
    train_sampler = TwoStreamBatchSampler(
        primary_indices=labeled_indices,
        secondary_indices=unlabeled_indices,
        batch_size=args.batch_size,
        secondary_batch_size=args.batch_size-args.labeled_bs,
    )

    train_loader = data.DataLoader(
        train_ds,
        # batch_size=args.batch_size,
        # shuffle=(train_sampler is None),
        num_workers=args.workers,
        batch_sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
    )
    val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    val_ds = data.Dataset(data=val_files, transform=val_transform)

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        # sampler=val_sampler,
        pin_memory=True,
        persistent_workers=True,
    )

    return [train_loader, val_loader]

