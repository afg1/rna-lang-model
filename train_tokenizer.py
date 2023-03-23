#!/usr/bin/env python
from dataloader import FASTALoader

from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer
from tokenizers.normalizers import Lowercase
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
import click

@click.group()
def cli():
    """
    This is the script for training the tokenizer
    """

@cli.command("train")
@click.argument("input_fasta")
@click.argument("output_path")
def main(input_fasta, output_path):
    print("Training tokenizer with chunks of 100k sequences")
    f_loader = FASTALoader(input_fasta)

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer=Lowercase()

    print("Using WordPiece tokenixer with vocab size 32k")
    trainer = WordPieceTrainer(
    vocab_size=32_000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    print("Training loop begins...")
    tokenizer.train_from_iterator(f_loader.get_some(n=100_000), trainer, length=len(f_loader))

    print("Finished!")
    tokenizer.save(output_path + "rna_tok", pretty=True)


if __name__ == "__main__":
    main()
