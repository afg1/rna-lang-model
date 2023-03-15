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
    def __init__(self, fasta_path):
        self.fasta_path = fasta_path
        print("Indexing FASTA file, may take a while...")
        self.record_dict = SeqIO.index(fasta_path, "fasta")
        self.iterator = iter(self.record_dict.items())
    
    def __len__(self):
        return len(self.record_dict)
    
    def __next__(self) -> str:
        _id, seq = next(self.iterator)

        # Here we see if we can augment the sequence or something...


        return str(seq.seq)

    def __iter__(self):
        return self


    def get_some(self, n=100_000):
        for i in range(0, len(self), n):
            ret_list = []
            for i in range(n):
                ret_list.append(next(self))

            yield ret_list

if __name__ == "__main__":
    # t = FASTALoader("../../data/rnacentral_active.fasta")
    #
    # s0 = next(t)
    #
    # print(s0)
    parser = FASTALoader("test.fasta")

    print(next(parser))
    print(parser.get_some(n=3))
