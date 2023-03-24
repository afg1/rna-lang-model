#!/usr/bin/env python3
from transformers import DistilBertConfig, DistilBertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import click

from rna_lang_model.data.FASTADataset import FASTADataset
from rna_lang_model.tokenizers import simple, trained

@click.command()
@click.option("--tokenizer", type=click.Choice(['simple', 'trained'], case_sensitive=False), default="simple")
@click.option("--tokenizer_file", default=None)
@click.option("--fasta_file", default=None)
@click.option("--mlm_prob", default = 0.15)
@click.option("--context_size", default=512)
@click.option("--output_dir", default=".")
@click.option("--use_fp16", is_flag=True, default=False, show_default=True)
@click.option("--overwrite_output", is_flag=True, default=True, show_default=True)
@click.option("--num_epochs", default=25)
@click.option("--batch_size", default=16)
@click.option("--seed", default=42)
@click.option("--use_mps", is_flag=True, default=False, show_default=True)
def main(tokenizer, tokenizer_file, fasta_file, mlm_prob, context_size, output_dir, overwrite_output, num_epochs, batch_size, use_fp16, seed, use_mps):
    if tokenizer == "trained":
        f_tokenizer = trained.get_tokenizer(tokenizer_file)
    else:
        f_tokenizer = simple.get_tokenizer()
    vocab_size = len(f_tokenizer.get_vocab())

    dataset = FASTADataset(fasta_file, f_tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=f_tokenizer, mlm=True, mlm_probability=mlm_prob
    )

    config = DistilBertConfig(vocab_size=vocab_size, max_position_embeddings=context_size)
    model = DistilBertForMaskedLM(config=config)



    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        seed=seed,
        fp16=use_fp16,
        use_mps_device=use_mps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()

    model.save_pretrained(output_dir)




if __name__ == "__main__":
    main()