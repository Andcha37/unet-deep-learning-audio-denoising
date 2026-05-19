import torch
from torch.utils.data import DataLoader, random_split
from dataset_utils.dataset import FullDataset
import copy

def get_dataloaders(clean_dir, noise_dir, batch_size=16, num_workers=4, split_ratio=(0.8, 0.1, 0.1), worker_init_fn=None, generator=None):
    fulldataset = FullDataset(
        clean_seg_dir = clean_dir,
        noise_dir = noise_dir,
        is_train = True
    )

    dataset_size = len(fulldataset)
    train_size = int(split_ratio[0] * dataset_size)
    val_size = int(split_ratio[1] * dataset_size)
    test_size = dataset_size - train_size - val_size 

    print("데이터셋 분할 완료")
    print(f"총 데이터 개수: {dataset_size}")
    print(f"train: {train_size} 개 / val: {val_size} 개 / test: {test_size} 개 \n")

    train_dataset, val_dataset, test_dataset = random_split(
        fulldataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 시드 고정
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    val_dataset.dataset = copy.copy(fulldataset)
    val_dataset.dataset.is_train = False

    test_dataset.dataset = copy.copy(fulldataset)
    test_dataset.dataset.is_train = False

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    return train_loader, val_loader, test_loader
