from torch.utils.data import Dataset
from torchvision.transforms import Compose


__all__ = [
    "FastDataset",
]


class FastDataset(Dataset):
    def __init__(
            self,
            data: Dataset,
            transform: Compose = None,
            post_transform: Compose = None,
            target_transform: Compose = None,
            post_target_transform: Compose = None,
            ):
        self.__data = data
        self._data = []
        self.transform = transform
        self.post_transform = post_transform
        self.target_transform = target_transform
        self.post_target_transform = post_target_transform
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
        X, y = self.data[idx]
        if self.post_transform:
            X = self.post_transform(X)
        if self.post_target_transform:
            y = self.post_target_transform(y)
        return (X, y)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return self._len

    @property
    def data(self):
        return self._data
