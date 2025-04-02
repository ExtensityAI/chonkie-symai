from symai import Import

ChonkieChunker = Import.load_expression('ExtensityAI/chonkie-symai', 'ChonkieChunker')

# Initialize chunker and tokenizer
chunker = ChonkieChunker(
    tokenizer_name="gpt2",
    chunker_name="RecursiveChunker"
)

print(chunker("Hello, world! This is a test."))


