import config
from sortedcontainers import SortedSet
import string
import numpy as np

from .config import Configurable, State


class Charset(Configurable):
    corups = State(default=string.hexdigits)
    blank = State(default=0)
    unknown = State(default=1)
    blank_char = State('\t')
    unknown_char = State('\n')
    case_sensitive = State(default=False)

    def __init__(self, corups=None, cmd={}, **kwargs):
        self.load_all(**kwargs)

        self._corpus = SortedSet(self._filter_corpus(corups))
        if self.blank_char in self._corpus:
            self._corpus.remove(self.blank_char)
        if self.unknown_char in self._corpus:
            self._corpus.remove(self.unknown_char)
        self._charset = list(self._corpus)
        self._charset.insert(self.blank, self.blank_char)
        self._charset.insert(self.unknown, self.unknown_char)
        self._charset_lut = {char: index
                             for index, char in enumerate(self._charset)}

    def _filter_corpus(self, corups):
        return corups

    def __getitem__(self, index):
        return self._charset[index]

    def index(self, x):
        target = x
        if not self.case_sensitive:
            target = target.upper()
        return self._charset_lut.get(target, self.unknown)

    def is_empty(self, index):
        return index == self.blank or index == self.unknown

    def is_empty_char(self, x):
        return x == self.blank_char or x == self.unknown_char

    def __len__(self):
        return len(self._charset)

    def string_to_label(self, string_input, max_size=32):
        length = max(max_size, len(string_input))
        target = np.zeros((length, ), dtype=np.int32)
        for index, c in enumerate(string_input):
            value = self.index(c)
            target[index] = value
        return target

    def label_to_string(self, label):
        ingnore = [self.unknown, self.blank]
        return "".join([self._charset[i] for i in label if i not in ingnore])


class ChineseCharset(Charset):

    def __init__(self, cmd={}, **kwargs):
        with open('./assets/chinese_charset.dic') as reader:
            corups = reader.read().strip()
        super().__init__(corups, cmd, **kwargs)

    def _filter_corpus(self, iterable):
        corups = []
        for char in iterable:
            if not self.case_sensitive:
                char = char.upper()
            corups.append(char)
        return corups


class ChineseOnlyCharset(ChineseCharset):
    def __init__(self, cmd={}, **kwargs):
        self.upper_range = (ord('A'), ord('Z'))
        self.lower_range = (ord('a'), ord('z'))
        super().__init__(cmd, **kwargs)

    def _filter_corpus(self, iterable):
        corups = []
        for char in iterable:
            if not self.is_english(char):
                corups.append(char)
        return corups

    def is_english(self, char):
        def between(x, lower, higher):
            return lower <= x <= higher
        return between(ord(char), *self.lower_range) or\
            between(ord(char), *self.upper_range)


class EnglishCharset(Charset):
    def __init__(self, cmd={}, **kwargs):
        corups = string.digits + string.ascii_uppercase
        super().__init__(corups, cmd, **kwargs)


class EnglishPrintableCharset(Charset):
    def __init__(self, cmd={}, **kwargs):
        corups = string.digits + string.ascii_letters + string.punctuation
        super().__init__(corups, cmd, **kwargs)


DefaultCharset = EnglishCharset
