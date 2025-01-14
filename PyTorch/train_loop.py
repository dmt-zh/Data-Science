import numpy as np
import struct
import shutil
import torch
import torchvision

from array import array
from collections.abc import Callable
from pathlib import Path
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2
from tqdm import tqdm
from typing import Self

device = 'cuda' if torch.cuda.is_available() else 'cpu'

######################################################################################

class MNISTDataset(Dataset):
    """MNISTDataset.

    Класс для создания тренировочного и тестового датасета
    на основе данных MNIST из модуля torchvision.
    """

    def __init__(
        self,
        path: Path,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None
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

    def train_data(self, exist: bool = False) -> Self:
        """Функция для создания тренировочного датасета."""

        if exist:
            self._create_dataset(self.path.joinpath('train'))
            return self
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

    def test_data(self, exist: bool = False) -> Self:
        """Функция для создания тестового датасета."""

        if exist:
            self._create_dataset(self.path.joinpath('test'))
            return self
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

######################################################################################

class MNISTDigitsClassifier(torch.nn.Module):
    """MNISTDigitsClassifier.

    Класс для создания и обучения модели классификации
    изображений на основе датасета MNIST.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Метод прямого прохода через слои классификатора."""

        return self.layer(batch)

######################################################################################

class Trainer:
    """Trainer.

    Класс для обучения и валидации модели
    классификатора цифр по изображениям MNIST.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int,
        lr: float,
        lr_patience: int,
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.compute_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=lr_patience)

    def train_model(self, epoch: int, train_loader: DataLoader, data_size: int) -> None:
        """Функция для обучения модели."""

        model.train()
        train_loss = []
        true_answer = 0
        training_loop = tqdm(train_loader, leave=False)
        for features, targets in training_loop:
            batch = features.reshape(-1, 784).to(device)
            targets = targets.reshape(-1)
            targets = torch.eye(10)[targets].to(torch.float32).to(device)
            predictions = self.model(batch)
            loss = self.compute_loss(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            mean_loss = sum(train_loss) / len(train_loss)
            true_answer += (predictions.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
            training_loop.set_description(
                f'Running epoch [{(epoch + 1)} / {self.epochs}], train_loss = {mean_loss:.4f}'
            )

        train_acc = true_answer / data_size
        print(f'Train true answer = {true_answer}')
        print(f'len(train_loader) = {len(train_loader)}')
        print(f'Epoch [{(epoch + 1)} / {self.epochs}]: Train loss = {mean_loss:.4f}; Train accuracy = {train_acc:.2f}')

    def eval_model(self, epoch: int, valid_loader: DataLoader, data_size: int):
        """Функция для валидации модели."""

        model.eval()
        with torch.no_grad():
            val_loss = []
            true_answer = 0
            for features, targets in tqdm(valid_loader, desc='Running evaluation', leave=False):
                batch = features.reshape(-1, 784).to(device)
                targets = targets.reshape(-1)
                targets = torch.eye(10)[targets].to(torch.float32).to(device)
                predictions = self.model(batch)
                loss = self.compute_loss(predictions, targets)
                val_loss.append(loss.item())
                mean_loss = sum(val_loss) / len(val_loss)
                true_answer += (predictions.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

            valid_acc = true_answer / data_size
            print(f'Epoch [{(epoch + 1)} / {self.epochs}]: Validation loss = {mean_loss:.4f}; Validation accuracy = {valid_acc:.2f}')

    def run(self, dataset: Dataset, batch_size: int) -> None:
        """Функция для запуска тренировщика."""

        train_data, val_data = random_split(train_dataset, [0.7, 0.3])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        for epoch in range(self.epochs):
            self.train_model(epoch, train_loader, len(train_data))
            self.eval_model(epoch, val_loader, len(val_data))

######################################################################################

transform_data = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5,), std=(0.5,))
    ]
)

######################################################################################

train_dataset = MNISTDataset('data/mnist', transform_data).train_data(exist=True)
test_data = MNISTDataset('data/mnist', transform_data).test_data(exist=True)
model = MNISTDigitsClassifier(784, 10).to(device)
trainer = Trainer(
    model=model,
    epochs=1,
    lr=0.005,
    lr_patience=5
)
trainer.run(
    dataset=train_dataset,
    batch_size=128,
)
