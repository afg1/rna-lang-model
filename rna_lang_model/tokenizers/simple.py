import tokenizers
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from tokenizers.normalizers import Lowercase
from tokenizers.decoders import BPEDecoder

"""
Implements the simplest possible tokenizer - one that encodes each nucleotide with its own ID

This should be everything needed to have the full MLM workflow later as well, but we shall see...
"""

def get_tokenizer():
    simple = tokenizers.Tokenizer(tokenizers.models.BPE(vocab={"a": 0, "t": 1, "g": 2, "c": 3, "n": 4, "[UNK]": 5, "[MASK]":6, "[PAD]": 7, "[CLS]":8, "[SEP]":9}, merges=[], unk_token="[UNK]"))
    simple.normalizer = Lowercase()
    simple.decoder = BPEDecoder()

    f_tokenizer = PreTrainedTokenizerFast(tokenizer_object=simple, unk_token="[UNK]")
    f_tokenizer.mask_token = "[MASK]"
    f_tokenizer.pad_token = "[PAD]"
    f_tokenizer.unk_token = "[UNK]"

    print(len(f_tokenizer.get_vocab()))
    return f_tokenizer



if __name__ == "__main__":
    t = get_tokenizer()
    print(t("aaaa"))