import redis


class RedisMeta:
    def __init__(self, socket_path):
        redis.StrictRedis()
