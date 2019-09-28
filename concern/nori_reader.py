import config


def NoriReader(paths=[]):
    import nori2 as nori
    from nori2.multi import MultiSourceReader
    if config.community_version:
        return MultiSourceReader(paths)
    else:
        return nori.Fetcher(paths)
