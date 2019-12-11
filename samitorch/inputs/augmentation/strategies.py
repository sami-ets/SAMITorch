from torchvision.transforms import Compose


class DataAugmentationStrategy(object):
    def __init__(self, transform: Compose = None, **kwargs):
        self._transform = transform

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def apply(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class AugmentInput(DataAugmentationStrategy):

    def __init__(self, transform: Compose):
        super().__init__(transform)

    def __call__(self, X):
        return self.apply(X)

    def apply(self, X):
        return self._transform(X)

    def reset(self):
        pass


class AugmentDuplicatedInput(DataAugmentationStrategy):
    """
    Apply a transformation to duplicate data with high probability and apply a transformation with low probability to
    unseen data
    """

    def __init__(self, transform: Compose):
        super().__init__(transform)
        self._seen_inputs_id = []

    def __call__(self, input_id, X):
        if input_id in self._seen_inputs_id:
            X = self.apply(X)
        else:
            self._seen_inputs_id.append(input_id)

        return X

    def apply(self, X):
        return self._transform(X)

    def reset(self):
        self._seen_inputs_id = []
