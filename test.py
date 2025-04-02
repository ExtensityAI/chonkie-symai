from symai import Import, Symbol

ChonkieChunker = Import.load_expression('ExtensityAI/chonkie-symai', 'ChonkieChunker')
# from src.chonkie_chunker import ChonkieChunker

# Initialize chunker and tokenizer
chunker = ChonkieChunker(
    tokenizer_name="gpt2",
    chunker_name="RecursiveChunker"
)

print(chunker(Symbol("Hello, world! This is a test.")))


