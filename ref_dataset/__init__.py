from .transform import get_transform as get_transform_seg
from .refdataset import Refdataset, collate_fn

def build_dataset(is_train, args, split='val'):
    print(is_train, split)
    data_path = args.data_path
    if is_train:
        split = 'train'
    else:
        split = split
    if args.data_set == 'refcoco':
        dataset = Refdataset(refer_data_root=data_path, dataset='refcoco', splitBy='unc', split=split, image_transforms=get_transform_seg((args.input_size, args.input_size), train=is_train))
    elif args.data_set == 'refcoco+':
        dataset = Refdataset(refer_data_root=data_path, dataset='refcoco+', splitBy='unc', split=split, image_transforms=get_transform_seg((args.input_size, args.input_size), train=is_train))
    elif args.data_set == 'refcocog':
        dataset = Refdataset(refer_data_root=data_path, dataset='refcocog', splitBy='umd', split=split, image_transforms=get_transform_seg((args.input_size, args.input_size), train=is_train))
    elif args.data_set == 'refzom':
        if split == 'val':
            split = 'test'
        dataset = Refdataset(refer_data_root=data_path, dataset='refzom', splitBy='zom', split=split, image_transforms=get_transform_seg((args.input_size, args.input_size), train=is_train))
    else:
        raise

    return dataset
