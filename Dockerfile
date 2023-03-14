FROM python:3.7-buster

RUN pip3 install biopython numpy scipy click transformers tokenizers

RUN mkdir /workdir

WORKDIR /workdir

COPY dataloader.py /workdir/dataloader.py
COPY train_tokenizer.py /workdir/train_tokenizer.py

ENTRYPOINT [ "/workdir/train_tokenizer.py" ]