from transformers import PreTrainedTokenizerFast


def get_tokenizer(path):
    f_tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
    f_tokenizer.mask_token = "[MASK]"
    f_tokenizer.pad_token = "[PAD]"

    return f_tokenizer


if __name__ == "__main__":
    t = get_tokenizer("rna_tok/sample1_32k_rna_tok")
    print(t("AAGCAA"))