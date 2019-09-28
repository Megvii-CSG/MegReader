import pickle
import hashlib
import os
import io
import time
import urllib.parse as urlparse
import warnings
import boto3
from botocore.exceptions import ClientError

from concern.config import Configurable, State
from concern.distributed import is_main
from concern.redis_meta import RedisMeta
import config
import redis


class MetaCache(Configurable):
    META_FILE = 'meta_cache.pickle'
    client = State(default='all')

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def cache(self, nori_path, meta=None):
        if meta is None:
            return self.read(nori_path)
        else:
            return self.save(nori_path, meta)

    def read(self, nori_path):
        raise NotImplementedError

    def save(self, nori_path, meta):
        raise NotImplementedError


class FileMetaCache(MetaCache):
    storage_dir = State(default=os.path.join(config.db_path, 'meta_cache'))
    inplace = State(default=not config.will_use_nori)

    def __init__(self, storage_dir=None, cmd={}, **kwargs):
        super(FileMetaCache, self).__init__(cmd=cmd, **kwargs)

        self.debug = cmd.get('debug', False)

    def ensure_dir(self):
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def storate_path(self, nori_path):
        if self.inplace:
            return os.path.join(
                    os.path.dirname(nori_path),
                    'meta_cache-%s.pickle' % self.client)
        return os.path.join(self.storage_dir, self.hash(nori_path) + '.pickle')

    def hash(self, nori_path: str):
        return hashlib.md5(nori_path.encode('utf-8')).hexdigest() + '-' + self.client

    def read(self, nori_path):
        file_path = self.storate_path(nori_path)
        if not os.path.exists(file_path):
            warnings.warn(
                'Meta cache not found: ' + file_path)
            warnings.warn('Now trying to read meta from scratch')
            return None
        with open(file_path, 'rb') as reader:
            try:
                return pickle.load(reader)
            except EOFError as e:  # recover from broken file
                if self.debug:
                    raise e
                return None

    def save(self, nori_path, meta):
        self.ensure_dir()

        with open(self.storate_path(nori_path), 'wb') as writer:
            pickle.dump(meta, writer)
        return True


class OSSMetaCache(MetaCache):
    HOST = State(default=config.oss_host)

    def __init__(self, **kwargs):
        super(OSSMetaCache, self).__init__(**kwargs)

    def oss_client(self):
        return boto3.client('s3', endpoint_url=self.HOST)

    def parse(self, nori_path):
        parse = urlparse.urlparse(nori_path)
        assert parse.scheme == 's3'

        bucket = parse.netloc
        path = os.path.join(parse.path, self.client, self.META_FILE)
        if path.startswith('/'):
            path = path[1:]

        return bucket, path

    def read(self, nori_path):
        bucket, path = self.parse(nori_path)
        try:
            resp = self.oss_client().get_object(Bucket=bucket, Key=path)
            data = resp['Body'].read()
        except ClientError as e:
            warnings.warn(
                'Get error when loading meta cache from oss:' + str(e))
            warnings.warn(
                'Nori path:' + str(nori_path))
            warnings.warn('Now trying to read meta from scratch')
            return None
        return pickle.loads(data)

    def save(self, nori_path, meta):
        data = io.BytesIO(pickle.dumps(meta))
        bucket, path = self.parse(nori_path)
        try:
            resp = self.oss_client().upload_fileobj(data, bucket, path)
            if resp is None:
                return True
        except ClientError as e:
            warnings.warn('Get error when loading meta from oss:' + e.message)
            return False
        return True


class RedisMetaCache(MetaCache):
    host = State(default=config.redis_host)
    port = State(default=config.redis_port)

    client = State(default='all')

    def __init__(self, **kwargs):
        super(RedisMetaCache, self).__init__(**kwargs)
        self.connection = redis.Redis(host=self.host, port=self.port)

    def hash(self, nori_path):
        return os.path.join(nori_path, self.client)

    def read(self, nori_path):
        path = self.hash(nori_path)
        while not self.connection.exists(os.path.join(path, '__keys__')):
            warnings.warn(
                'Cache not found in redis:' + str(path))
            if is_main():
                warnings.warn('Now may try to load and save meta from file')
                return None
            else:
                time.sleep(2)
        return RedisMeta(self.connection, path)

    def save(self, nori_path, meta):
        if not is_main():
            return False

        path = self.hash(nori_path)
        for key, value_list in meta.items():
            pipe = self.connection.pipeline()
            pipe.sadd(os.path.join(path, '__keys__'), key)
            for value in value_list:
                pipe.lpush(os.path.join(path, key), value)
            pipe.execute()
        return True
