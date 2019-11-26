import config

assert not (config.will_use_nori and config.will_use_lmdb), 'only one metaloader can be used'
if config.will_use_nori:
    from .nori_meta_loader import NoriMetaLoader
    MetaLoader = NoriMetaLoader
elif config.will_use_lmdb:
    from .lmdb_meta_loader import LMDBMetaLoader
    MetaLoader = LMDBMetaLoader
else:
    from .json_meta_loader import JsonMetaLoader
    MetaLoader = JsonMetaLoader
