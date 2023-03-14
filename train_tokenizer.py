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
    f_loader = FASTALoader(input_fasta, limit=100)

    # tokenizer = ByteLevelBPETokenizer()
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    # tokenizer.train_from_iterator(f_loader, vocab_size=64_000, min_frequency=10, special_tokens=[
    #     "<s>",
    #     "<pad>",
    #     "</s>",
    #     "<unk>",
    #     "<mask>"
    # ])

    trainer = WordPieceTrainer(
    vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    tokenizer.train_from_iterator(f_loader, trainer)

    tokenizer.save(output_path + "rna_tok", pretty=True)


if __name__ == "__main__":
    cli()
