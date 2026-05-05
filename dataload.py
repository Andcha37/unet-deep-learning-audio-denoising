import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split

class AudioDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform

        self.lst_input = sorted(os.listdir(self.noisy_dir))

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):
        file_name = self.lst_input[index]

        input_mag = torch.load(os.path.join(self.noisy_dir, file_name))
        label_mag = torch.load(os.path.join(self.clean_dir, file_name))

        if input_mag.ndim == 2:
            input_mag = input_mag.unsqueeze(0)
        if label_mag.ndim == 2:
            label_mag = label_mag.unsqueeze(0)

        data = {'input': input_mag, 'label': label_mag}

        if self.transform:
            data = self.transform(data)

        return data

def get_audio_loaders(noisy_dir, clean_dir, batch_size=16, split_ratio=(0.8, 0.1, 0.1)):
    full_dataset = AudioDataset(noisy_dir, clean_dir)
    
    dataset_size = len(full_dataset)
    train_size = int(split_ratio[0] * dataset_size)
    val_size = int(split_ratio[1] * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"현재 사용 장치 cpu vs gpu : {device}")

full_dataset = AudioDataset(
    noisy_dir=r"믹스 음성 파일 디렉토리",
    clean_dir=r"클린 음성 파일 디렉토리"
)

total_size = len(full_dataset)
train_size = int(len(full_dataset) * 0.8)
val_size = int(len(full_dataset) * 0.1)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)
'''

'''
for batch in train_loader:
    print(f"Input batch shape: {batch['input'].shape}") # 기대값: [16, 1, H, W]
    print(f"Label batch shape: {batch['label'].shape}")
    break
'''