import tokenizers
from tokenizers.pre_tokenizers import Split, PreTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from tokenizers.normalizers import Lowercase
from tokenizers.decoders import BPEDecoder, WordPiece
import itertools as it
import re
from typing import List

class CustomPretokenizer:
    def __init__(self, k):
        self.regex = f'.{{1,{k}}}'

    def kmer_split(self, i: int, normalized_string: tokenizers.NormalizedString) -> List[tokenizers.NormalizedString]:
        
        return [normalized_string[m.start(0):m.end(0)] for m in re.finditer(self.regex, str(normalized_string)) ]

    def pre_tokenize(self, pretok: tokenizers.PreTokenizedString):
        pretok.split(self.kmer_split)

def generate_vocab(k):
    """
    Generate the vocab dict for the given kmer length
    """
    kmer_generator = it.chain.from_iterable([it.product('atgc', repeat=n) for n in range(1, k+1)])

    ## Make kmer entries in vocab
    vocab = {"".join(kmer) : n for n, kmer in enumerate(kmer_generator)}
    
    ## Add special tokens
    vocab["[UNK]"] = len(vocab) 
    vocab["[MASK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    vocab["[CLS]"] = len(vocab)
    vocab["[SEP]"] = len(vocab)
    
    return vocab



def get_tokenizer(k=3):
    vocab = generate_vocab(k)
    simple = tokenizers.Tokenizer(tokenizers.models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    simple.pre_tokenizer = PreTokenizer.custom(CustomPretokenizer(k)) #Split(r"g{1,3}", "merged_with_next", invert=True)
    simple.normalizer = Lowercase()
    simple.mask_token = "[MASK]"
    simple.pad_token = "[PAD]"
    simple.unk_token = "[UNK]"
    simple.decoder = WordPiece()
    simple.post_processor = TemplateProcessing(
        single="[CLS] $0 [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 0)],
    )

    return simple



if __name__ == "__main__":
    t = get_tokenizer(k=5)
    print(t.encode('ACTGCGG').tokens)
    ## Check it can build a data collator
    from transformers import DataCollatorForLanguageModeling
    dc = DataCollatorForLanguageModeling(t)