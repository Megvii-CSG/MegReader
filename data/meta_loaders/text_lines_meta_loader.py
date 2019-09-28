from data.text_lines import TextLines
from .meta_loader import MetaLoader


class TextLinesMetaLoader(MetaLoader):
    def parse_meta(self, data_id, meta):
        return dict(
                data_id=data_id,
                lines=TextLines(self.get_annotation(meta)['lines']))
