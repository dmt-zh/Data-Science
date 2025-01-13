import numpy as np
import struct
import shutil
import torchvision

from array import array
from collections.abc import Callable
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2

from pathlib import Path
from torch import Tensor
from tqdm import tqdm
from typing import Self
from PIL import Image


class MNISTDataset(Dataset):
    """MNISTDataset.

    Класс для создания тренировочного и тестового датасета
    на основе данных MNIST из модуля torchvision.
    """

    def __init__(
        self,
        path: Path,
        transform: Callable[[Tensor], Tensor] | None = None
    ) -> None:
        self.path = Path(path).resolve()
        self.transform = transform
        self.classes = None
        self.class_to_idx = None
        self._data = []
        self._dataset_length = 0

    def __len__(self) -> int:
        """Метод возвращает длину датасета."""

        return self._dataset_length

    def __getitem__(self, index: int) -> tuple[Image, int]:
        """Метод возвращает кортеж из изображения и цифры
        указанной на изображении по заданному индексу.
        """

        image_path, target = self._data[index]
        image = np.array(Image.open(image_path))
        if self.transform:
            image = self.transform(image)
        return image, target

    def _create_dataset(self, path: Path) -> None:
        """Функция для формирования финального датасета."""

        self.classes = list(path.name for path in sorted(path.iterdir()))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        for image_path in tqdm(path.glob('*/*.jpg'), desc=f'Creating "{path.name.upper()}" data'):
            data_class = self.class_to_idx[image_path.parent.name]
            self._data.append((image_path, data_class))
            self._dataset_length += 1

    def _download_mnist_data(self, download_path: Path, train: bool = True) -> Path:
        """Функция для скачивания датасета MNIST."""

        torchvision.datasets.MNIST(
            root=download_path,
            train=train,
            download=True,
        )
        return download_path.joinpath('MNIST/raw')

    def _read_and_write(
        self,
        root: Path,
        features: Path,
        labels: Path,
        size: int,
        rows: int,
        cols: int,
        prefix: str,
    ) -> Path:
        """Функция для считывания сырых данных и формирования
        изображений по директориям соответствующих классов."""

        data_path = root.joinpath(prefix)
        data_classes = {digit: f'class_{digit}' for digit in range(10)}
        image_data = array('B', features.read_bytes()[16:])
        labels_sequence = array('b', labels.read_bytes()[8:])

        for idx, label in tqdm(enumerate(labels_sequence), desc=f'Processing raw "{prefix.upper()}" data'):
            class_dir = data_path.joinpath(data_classes[label])
            class_dir.mkdir(exist_ok=True, parents=True)
            image_path = data_path.joinpath(class_dir, f'{str(idx)}.jpg')
            image_as_array = np.asarray([
                image_data[(idx * rows * cols + num * cols): (idx * rows * cols + (num + 1) * cols)] for num in range(rows)
            ])
            image = Image.fromarray(image_as_array)
            image.save(image_path)
        return data_path

    def train_data(self) -> Self:
        """Функция для создания тренировочного датасета."""

        raw_data_path = self._download_mnist_data(self.path)
        features = next(raw_data_path.glob('train-images*ubyte'))
        labels = next(raw_data_path.glob('train-labels*ubyte'))
        _, size, rows, cols = struct.unpack('>IIII', features.read_bytes()[:16])
        data_path = self._read_and_write(
            root=self.path,
            features=features,
            labels=labels,
            size=size,
            rows=rows,
            cols=cols,
            prefix='train',
        )
        self._create_dataset(data_path)
        shutil.rmtree(raw_data_path.parent, ignore_errors=True)
        return self

    def test_data(self) -> Self:
        """Функция для создания тестового датасета."""

        raw_data_path = self._download_mnist_data(self.path, train=False)
        features = next(raw_data_path.glob('t10k-images*ubyte'))
        labels = next(raw_data_path.glob('t10k-labels*ubyte'))
        _, size, rows, cols = struct.unpack('>IIII', features.read_bytes()[:16])
        data_path = self._read_and_write(
            root=self.path,
            features=features,
            labels=labels,
            size=size,
            rows=rows,
            cols=cols,
            prefix='test',
        )
        self._create_dataset(data_path)
        shutil.rmtree(raw_data_path.parent, ignore_errors=True)
        return self

MNIST_DATA_PATH = 'data/mnist'
train_dataset = MNISTDataset(MNIST_DATA_PATH).train_data()
test_data = MNISTDataset(MNIST_DATA_PATH).test_data()
