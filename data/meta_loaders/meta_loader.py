import config

if config.will_use_nori:
    from .nori_meta_loader import NoriMetaLoader
    MetaLoader = NoriMetaLoader
else:
    from .json_meta_loader import JsonMetaLoader
    MetaLoader = JsonMetaLoader
