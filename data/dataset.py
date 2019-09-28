import glob
from torch.utils.data import Dataset as TorchDataset

from concern.config import Configurable, State


class Dataset(TorchDataset, Configurable):
    r'''Dataset reading data.
    Args:
        meta_loader: Instance of Metaloader, which is defined in data/meta_loaders/meta_loader.py,
            for decoding annotation from packed files.
        unpack: Callable instance which unpack data from packed bytes such as pickle or msgpack.
            If not provided, the `default_unpack` will be invoked.
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    meta_loader = State()
    unpack = State(default=None)
    split = State(default=1)

    processes = State(default=[])

    def prepare_meta_single(self, path_name):
        return self.meta_loader.load_meta(path_name)

    def list_or_pattern(self, path):
        if isinstance(path, str):
            if '*' in path:
                return glob.glob(path)
            else:
                return [path]
        else:
            return path

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples

        data_id = self.data_ids[index]
        meta = dict()

        for key in self.meta:
            meta[key] = self.meta[key][index]

        try:
            data = self.unpack(data_id, meta)
            if self.processes is not None:
                for data_process in self.processes:
                    data = data_process(data)
        except Exception as e:
            if self.debug or retry > 10:
                raise e
            return self.__getitem__(index + 1, retry=retry+1)
        return data

    def truncate(self, rank, total):
        clip_size = self.num_samples // total
        start = rank * clip_size
        if rank == total - 1:
            ending = self.num_samples
        else:
            ending = start + clip_size
        new_data_ids = self.data_ids[start:ending]
        del self.data_ids
        self.data_ids = new_data_ids

        new_meta = dict()
        for key in self.meta.keys():
            new_meta[key] = self.meta[key][start:ending]
        for key in new_meta.keys():
            del self.meta[key]
        self.meta = new_meta
        self.num_samples = len(self.data_ids)
        self.truncated = True

    def select(self, indices):
        if self.truncated:
            return
        new_data_ids = [self.data_ids[i] for i in indices]
        del self.data_ids
        self.data_ids = new_data_ids

        new_meta = {
            key: [self.meta[key][i] for i in indices] for key in self.meta.keys()
        }
        del self.meta
        self.meta = new_meta
        # self.num_samples = len(self.data_ids)
        self.truncated = True

    def __len__(self):
        if self.debug:
            return 512000
        return self.num_samples // self.split
