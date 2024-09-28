from torch.utils.data import Dataset
from torchvision.transforms import Compose


__all__ = [
    "FastDataset",
]


class FastDataset(Dataset):
    def __init__(self, data, transform: Compose = None, target_transform: Compose = None):
        self.__data = data
        self._data = []
        self.transform = transform
        self.target_transform = target_transform
        self._len = len(data)
        self._post_init()

    def _post_init(self):
        for i in range(len(self)):
            X, y = self.__data[i]
            if self.transform:
                X = self.transform(X)
            if self.target_transform:
                y = self.target_transform(y)
            self._data.append((X, y))
        self._data = tuple(self._data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return self._len

    @property
    def data(self):
        return self._data
