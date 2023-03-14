process download_fasta {
    queue 'short'
     
    output:
        path("*.fasta")

    """
    wget https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz
    gzip -d rnacentral_active.fasta.gz

    """
}

process train_tokenizer {
    container 'docker://afgreen/rna-lang-tok:latest'
    memory '32G'
    cpus 8

    input:
        path(fasta)
    
    output:
        path("rna_tok*")

    """
    train_tokenizer.py train $fasta rna_tok
    """
}

workflow {
    emit:
        tokenizer
    
    download_fasta | train_tokenizer | set { tokenizer }
}