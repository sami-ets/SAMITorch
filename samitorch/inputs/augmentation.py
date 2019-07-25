import abc

from torchvision.transforms import Compose


class DataAugmentationStrategy(metaclass=abc.ABCMeta):
    def __init__(self, transform: Compose) -> None:
        super().__init__()
        self._transform = transform

    @abc.abstractmethod
    def apply(self, X, y):
        raise NotImplementedError


class AugmentDuplicatedInstance(DataAugmentationStrategy):

    def __init__(self, transform: Compose) -> None:
        super().__init__(transform)
        self._seen_instance = []

    def apply(self, X, y):
        pass
