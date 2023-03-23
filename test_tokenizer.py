import click
from tokenizers import Tokenizer
from tokenizers.normalizers import Lowercase
from rna_lang_model.data.FASTALoader import FASTALoader
import pandas as pd
import matplotlib.pyplot as plt

@click.command()
@click.argument("tokenizer_json")
@click.argument("fasta")
def test_tokenizer(tokenizer_json, fasta):
    tokenizer = Tokenizer.from_file(tokenizer_json)
    tokenizer.normalizer=Lowercase()

    f_loader = FASTALoader(fasta)

    sequence_lengths = []
    token_lengths = []
    for batch in f_loader.get_some(10):
        batch_tokens = tokenizer.encode_batch(batch)
        for seq in batch:
            sequence_lengths.append(len(seq))
        for enc in batch_tokens:
            token_lengths.append(len(enc.ids))


    data = pd.DataFrame({"seq_len":sequence_lengths, "tok_len":token_lengths})
    data["ratio"] = data["tok_len"]/data["seq_len"]
    plt.hist(data["ratio"], bins=100)
    plt.xlabel("Compression factor")
    plt.ylabel("Count")
    plt.title("Compression factor for RNA BPE tokenizer")
    plt.savefig("BPETokenizer_compression.png")
    plt.show()





if __name__ == "__main__":
    test_tokenizer()
