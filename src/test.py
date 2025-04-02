from symai import Import

ChonkieChunker = Import.load_expression('ExtensityAI/chonkie-symai', 'ChonkieChunker')

def main():
    # Initialize chunker with default settings
    chunker = ChonkieChunker()

    # Test data
    test_text = "This is a test sentence. This is another sentence. And here's a third one!"
    print("\nTesting different chunker types:")
    print("-" * 50)

    # Test different chunker types
    chunker_types = ["TokenChunker", "SentenceChunker", "RecursiveChunker", "SemanticChunker", "SDPMChunker"]

    for chunker_type in chunker_types:
        print(f"\nTesting {chunker_type}:")
        chunks = chunker(test_text, chunker_name=chunker_type)
        print(f"Number of chunks: {len(chunks.value)}")
        print("Chunks:", chunks.value)

    # Test text cleaning
    print("\nTesting text cleaning:")
    print("-" * 50)
    dirty_text = "Hello\x00World! Invalid UTF-8: \xff"
    cleaned_text = ChonkieChunker.clean_text(dirty_text)
    print(f"Original text: {dirty_text}")
    print(f"Cleaned text: {cleaned_text}")

    # Test invalid chunker
    print("\nTesting invalid chunker:")
    print("-" * 50)
    try:
        chunker("Some text", chunker_name="NonExistentChunker")
    except ValueError as e:
        print(f"Successfully caught error: {str(e)}")

if __name__ == "__main__":
    main()



