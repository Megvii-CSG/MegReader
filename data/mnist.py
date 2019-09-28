import torchvision

from concern.config import Configurable, State


class MNistDataset(Configurable, torchvision.datasets.MNIST):
    root = State()
    is_train = State(autoload=False)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        self.is_train = cmd['is_train']

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
        ])
        torchvision.datasets.MNIST.__init__(
            self, self.root,
            train=self.is_train, download=True, transform=transform
        )
