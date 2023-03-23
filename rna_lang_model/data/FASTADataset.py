from rna_lang_model.data.FASTALoader import FASTALoader
from torch.utils.data import Dataset
import torch


class FASTADataset(Dataset):
    def __init__(self, fasta_file, tokenizer):
        loader = FASTALoader(fasta_file)
        self.data = []
        for seq in loader.get_some(10_000):
            self.data.extend(tokenizer(seq, truncation=True,
                                      max_length=512, 
                                      padding=False).input_ids)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)




# from tokenizers import Tokenizer
# from tokenizers.normalizers import Lowercase
# tokenizer = Tokenizer.from_file("sample1_32k_rna_tok")
# tokenizer.normalizer=Lowercase()
# t = FASTADataset("sample.fasta", tokenizer)

# print(t[1234])