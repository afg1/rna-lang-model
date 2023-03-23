#!/usr/bin/env python
from rna_lang_model.data.FASTALoader import FASTALoader

from tokenizers import Tokenizer
from tokenizers.normalizers import Lowercase
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder

import click


@click.command()
@click.argument("input_fasta")
@click.argument("output_path")
@click.option("--vocab_size", default=32_000)
@click.option("--chunk_size", default=100_000)
def main(input_fasta, output_path, vocab_size, chunk_size):
    print(f"Training tokenizer with chunks of {chunk_size} sequences")
    f_loader = FASTALoader(input_fasta)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer=Lowercase()
    tokenizer.decoder =  BPEDecoder()

    print(f"Using BPE tokenizer with vocab size {vocab_size}")
    trainer = BpeTrainer(
    vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    print("Training loop begins...")
    tokenizer.train_from_iterator(f_loader.get_some(n=chunk_size), trainer, length=len(f_loader))

    print("Finished!")
    tokenizer.save(output_path + "rna_tok.json", pretty=True)


if __name__ == "__main__":
    main()
