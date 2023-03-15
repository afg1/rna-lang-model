#!/usr/bin/env python
from dataloader import FASTALoader

from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer
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
    f_loader = FASTALoader(input_fasta, limit=100_000)

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    print("Using WordPiece tokenixer with vocab size 16k")
    trainer = WordPieceTrainer(
    vocab_size=16_384, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    print("Training loop begins...")
    tokenizer.train_from_iterator(f_loader, trainer)

    print("Finished!")
    tokenizer.save(output_path + "rna_tok", pretty=True)


if __name__ == "__main__":
    cli()
