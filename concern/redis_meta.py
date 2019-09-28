import os
import bisect

import redis


class RedisMeta:
    def __init__(self, connection, prefix, key=None):
        self.connection = connection
        self._key = key
        self.prefix = prefix

    def __getitem__(self, key):
        if self.key() is None:
            if self.connection.exists(self.key(key)):
                return type(self)(self.connection, prefix=self.prefix, key=key)
            raise KeyError
        return self.connection.lindex(self.key(), key).decode()

    def __contains__(self, key):
        if self.key() is None:
            assert isinstance(key, str)
            return self.connection.exists(self.key(key))
        else:
            assert isinstance(key, int)
            return key < len(self) and key > -len(self) + 1

    def __iter__(self):
        if self.key() is None:
            iter(self.keys)
        else:
            for i in range(len(self)):
                yield self.__getitem__(i)

    def __add__(self, another):
        return ConcateMetaRedisMeta(self, another)

    def get(self, key, default=None):
        assert self.key() is None

    def key(self, key=None):
        if key is None:
            key = self._key
        if key is None:
            return None
        return os.path.join(self.prefix, key)

    def keys(self):
        assert self.key() is None
        keys = self.connection.smembers(self.key('__keys__'))
        return {key.decode() for key in keys}

    def items(self):
        assert self.key() is None
        tuples = []
        for key in self.keys():
            tuples.append((key, self.__getitem__(key)))
        return tuples


    def __len__(self):
        if self.key() is None:
            return len(self.keys())
        return self.connection.llen(self.key())


class ConcateMetaRedisMeta:
    def __init__(self, *meta_list):
        milestones = []
        start = 0
        for meta in meta_list:
            assert meta.key() is not None
            start += len(meta)
            milestones.append(start)

        self.milestones = milestones
        self.num_samples = start
        self.meta_list = list(meta_list)

    def __getitem__(self, index):
        meta_index = bisect.bisect(self.milestones, index)
        if meta_index == 0:
            real_index = index
        else:
            real_index = index - self.milestones[meta_index - 1]
        return self.meta_list[meta_index][real_index]

    def __len__(self):
        return self.num_samples

    def __add__(self, another):
        self.num_samples += len(another)
        self.milestones.append(self.num_samples)
        self.meta_list.append(another)
        return self
