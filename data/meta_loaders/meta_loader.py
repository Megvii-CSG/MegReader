import config

if config.will_use_nori:
    from .nori_meta_loader import NoriMetaLoader
    MetaLoader = NoriMetaLoader
elif config.will_use_lmdb:
    from .lmdb_meta_loader import LMDBMetaLoader
    MetaLoader = LMDBMetaLoader
else:
    from .json_meta_loader import JsonMetaLoader
    MetaLoader = JsonMetaLoader
