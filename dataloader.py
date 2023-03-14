"""
Write a dataloader class to go directly from the FASTA file to something the
tokenizer/model can train from

I think it is ok to loop over the FASTA in order, i.e. from top to bottom. I
can't see aything in the transformers docs to suggest this won't work, but it
may not be ideal.

We need to identify the start and end of the sequence, then strip any whitespace
(probably newlines?).

Given this is FASTA, I think there will be some sequences with N in. These can
either be discarded, or used to randomly augment by replacing with a random
base.
 """
import io
from Bio import SeqIO

class FASTALoader():
    def __init__(self, fasta_path, limit=10_000):
        self.fasta_path = fasta_path
        self.parser = SeqIO.parse(fasta_path, "fasta")

    def __next__(self) -> str:
        seq = next(self.parser)

        # Here we see if we can augment the sequence or something...


        return seq.seq

    def __iter__(self):
        return self


    def get_some(self, n=100_000):
        ret_list = []
        for i in range(n):
            ret_list.append(next(self))

        return ret_list

if __name__ == "__main__":
    # t = FASTALoader("../../data/rnacentral_active.fasta")
    #
    # s0 = next(t)
    #
    # print(s0)
    parser = FASTALoader("test.fasta")

    print(next(parser))
