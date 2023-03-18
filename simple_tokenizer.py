import tokenizers
from tokenizers.processors import TemplateProcessing

"""
Implements the simplest possible tokenizer - one that encodes each nucleotide with its own ID

This should be everything needed to have the full MLM workflow later as well, but we shall see...
"""

tok = tokenizers.Tokenizer(tokenizers.models.BPE(vocab={"A": 1, "T": 2, "G": 3, "C": 4, "N": 5}, merges=[], unk_token="<unk>"))
tok.encode("AAACTGNN")
tok.add_special_tokens(["<cls>", "<sep>", "<mask>", "<pad>"])

tok.post_processor = TemplateProcessing(single="<cls> $A <sep>", pair="<cls> $A <sep> $B:1 <sep>:1", special_tokens=[("<cls>", 6), ("<sep>", 7)])
