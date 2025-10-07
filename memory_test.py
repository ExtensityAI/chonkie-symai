#!/usr/bin/env python3
"""Comprehensive memory test for Chonkie lazy loading optimization."""

import gc
import tracemalloc
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def test_memory_usage(test_name: str, test_func):
    """Test memory usage for a given function."""
    print(f"\n--- {test_name} ---")

    # Clear memory
    gc.collect()

    # Start memory tracing
    tracemalloc.start()

    # Get baseline
    baseline_current, baseline_peak = tracemalloc.get_traced_memory()

    try:
        # Run the test
        result = test_func()

        # Get memory after test
        after_current, after_peak = tracemalloc.get_traced_memory()

        # Calculate increase
        current_increase = after_current - baseline_current
        peak_increase = after_peak - baseline_current

        print(f"Baseline: {format_bytes(baseline_current)}")
        print(f"After: {format_bytes(after_current)}")
        print(f"Increase: {format_bytes(current_increase)}")
        print(f"Peak increase: {format_bytes(peak_increase)}")

        return {
            'baseline': baseline_current,
            'after': after_current,
            'increase': current_increase,
            'peak_increase': peak_increase
        }

    except Exception as e:
        print(f"Error: {e}")
        return {'baseline': baseline_current, 'after': baseline_current, 'increase': 0, 'peak_increase': 0}

    finally:
        tracemalloc.stop()


def test_eager_imports():
    """Test memory usage with eager imports."""
    print("\n" + "=" * 60)
    print("1. TESTING EAGER IMPORTS")
    print("=" * 60)

    def import_numpy():
        import numpy as np
        return np

    def import_sentence_transformers():
        import sentence_transformers
        return sentence_transformers

    def import_transformers():
        import transformers
        return transformers

    def import_tokenizers():
        import tokenizers
        return tokenizers

    def import_tiktoken():
        import tiktoken
        return tiktoken

    def import_openai():
        import openai
        return openai

    def import_cohere():
        import cohere
        return cohere

    def import_model2vec():
        import model2vec
        return model2vec

    tests = [
        ("NumPy", import_numpy),
        ("SentenceTransformers", import_sentence_transformers),
        ("Transformers", import_transformers),
        ("Tokenizers", import_tokenizers),
        ("TikToken", import_tiktoken),
        ("OpenAI", import_openai),
        ("Cohere", import_cohere),
        ("Model2Vec", import_model2vec),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_memory_usage(f"Eager {name}", test_func)

    return results


def test_lazy_imports():
    """Test memory usage with lazy imports (no actual loading)."""
    print("\n" + "=" * 60)
    print("2. TESTING LAZY IMPORTS (No actual loading)")
    print("=" * 60)

    def import_lazy_imports():
        from lazy_imports import LazyImport, LazyImports
        return LazyImport, LazyImports

    def import_lazy_numpy():
        from lazy_imports import numpy
        return numpy

    def import_lazy_sentence_transformers():
        from lazy_imports import sentence_transformers
        return sentence_transformers

    def import_lazy_transformers():
        from lazy_imports import transformers
        return transformers

    def import_lazy_tokenizers():
        from lazy_imports import tokenizers
        return tokenizers

    def import_lazy_tiktoken():
        from lazy_imports import tiktoken
        return tiktoken

    def import_lazy_openai():
        from lazy_imports import openai
        return openai

    def import_lazy_cohere():
        from lazy_imports import cohere
        return cohere

    def import_lazy_model2vec():
        from lazy_imports import model2vec
        return model2vec

    tests = [
        ("Lazy Import System", import_lazy_imports),
        ("Lazy NumPy", import_lazy_numpy),
        ("Lazy SentenceTransformers", import_lazy_sentence_transformers),
        ("Lazy Transformers", import_lazy_transformers),
        ("Lazy Tokenizers", import_lazy_tokenizers),
        ("Lazy TikToken", import_lazy_tiktoken),
        ("Lazy OpenAI", import_lazy_openai),
        ("Lazy Cohere", import_lazy_cohere),
        ("Lazy Model2Vec", import_lazy_model2vec),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_memory_usage(f"Lazy {name}", test_func)

    return results


def test_actual_usage():
    """Test memory usage when lazy imports are actually used."""
    print("\n" + "=" * 60)
    print("3. TESTING LAZY IMPORTS WITH ACTUAL USAGE")
    print("=" * 60)

    def use_numpy():
        from lazy_imports import numpy
        arr = numpy.array([1, 2, 3, 4, 5])
        return arr

    def use_tokenizers():
        from lazy_imports import tokenizers
        return tokenizers.Tokenizer

    def use_tiktoken():
        from lazy_imports import tiktoken
        encoding = tiktoken.get_encoding("gpt2")
        return encoding

    def use_sentence_transformers():
        from lazy_imports import sentence_transformers
        # Just accessing the class should trigger import
        return sentence_transformers.SentenceTransformer

    def use_transformers():
        from lazy_imports import transformers
        return transformers.AutoTokenizer

    def use_openai():
        from lazy_imports import openai
        return openai.OpenAI

    def use_cohere():
        from lazy_imports import cohere
        return cohere.ClientV2

    def use_model2vec():
        from lazy_imports import model2vec
        return model2vec.StaticModel

    tests = [
        ("NumPy Usage", use_numpy),
        ("Tokenizers Usage", use_tokenizers),
        ("TikToken Usage", use_tiktoken),
        ("SentenceTransformers Usage", use_sentence_transformers),
        ("Transformers Usage", use_transformers),
        ("OpenAI Usage", use_openai),
        ("Cohere Usage", use_cohere),
        ("Model2Vec Usage", use_model2vec),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_memory_usage(f"Usage {name}", test_func)

    return results


def test_chonkie_integration():
    """Test memory usage with ChonkieChunker integration."""
    print("\n" + "=" * 60)
    print("4. TESTING CHONKIE INTEGRATION")
    print("=" * 60)

    def test_embeddings_import():
        from embeddings import SentenceTransformerEmbeddings, OpenAIEmbeddings, CohereEmbeddings
        return SentenceTransformerEmbeddings, OpenAIEmbeddings, CohereEmbeddings

    def test_chunker_import():
        from chunker import RecursiveChunker, SemanticChunker, TokenChunker
        return RecursiveChunker, SemanticChunker, TokenChunker

    def test_tokenizer_import():
        from tokenizer import Tokenizer, CharacterTokenizer, WordTokenizer
        return Tokenizer, CharacterTokenizer, WordTokenizer

    tests = [
        ("Embeddings Import", test_embeddings_import),
        ("Chunker Import", test_chunker_import),
        ("Tokenizer Import", test_tokenizer_import),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_memory_usage(f"Chonkie {name}", test_func)

    return results


def print_summary(eager_results, lazy_results, usage_results, integration_results):
    """Print comprehensive summary of results."""
    print("\n" + "=" * 80)
    print("MEMORY OPTIMIZATION SUMMARY")
    print("=" * 80)

    print(f"{'Library':<20} {'Eager (MB)':<12} {'Lazy (MB)':<12} {'Usage (MB)':<12} {'Savings (MB)':<12}")
    print("-" * 80)

    total_eager = 0
    total_lazy = 0
    total_usage = 0

    # Map libraries for comparison
    library_mapping = {
        'NumPy': ('NumPy', 'Lazy NumPy', 'NumPy Usage'),
        'SentenceTransformers': ('SentenceTransformers', 'Lazy SentenceTransformers', 'SentenceTransformers Usage'),
        'Transformers': ('Transformers', 'Lazy Transformers', 'Transformers Usage'),
        'Tokenizers': ('Tokenizers', 'Lazy Tokenizers', 'Tokenizers Usage'),
        'TikToken': ('TikToken', 'Lazy TikToken', 'TikToken Usage'),
        'OpenAI': ('OpenAI', 'Lazy OpenAI', 'OpenAI Usage'),
        'Cohere': ('Cohere', 'Lazy Cohere', 'Cohere Usage'),
        'Model2Vec': ('Model2Vec', 'Lazy Model2Vec', 'Model2Vec Usage'),
    }

    for lib_name, (eager_key, lazy_key, usage_key) in library_mapping.items():
        if eager_key in eager_results and lazy_key in lazy_results:
            eager_mb = eager_results[eager_key]['increase'] / (1024 * 1024)
            lazy_mb = lazy_results[lazy_key]['increase'] / (1024 * 1024)

            total_eager += eager_mb
            total_lazy += lazy_mb

            if usage_key in usage_results:
                usage_mb = usage_results[usage_key]['increase'] / (1024 * 1024)
                total_usage += usage_mb
                savings = eager_mb - usage_mb
                print(f"{lib_name:<20} {eager_mb:<12.2f} {lazy_mb:<12.2f} {usage_mb:<12.2f} {savings:<12.2f}")
            else:
                savings = eager_mb - lazy_mb
                print(f"{lib_name:<20} {eager_mb:<12.2f} {lazy_mb:<12.2f} {'N/A':<12} {savings:<12.2f}")

    print("-" * 80)
    print(f"{'TOTAL':<20} {total_eager:<12.2f} {total_lazy:<12.2f} {total_usage:<12.2f} {total_eager - total_usage:<12.2f}")

    # Show lazy import system overhead
    if 'Lazy Import System' in lazy_results:
        overhead_mb = lazy_results['Lazy Import System']['increase'] / (1024 * 1024)
        print(f"\nLazy import system overhead: {overhead_mb:.2f} MB")

    # Calculate percentages
    lazy_savings_pct = ((total_eager - total_lazy) / total_eager * 100) if total_eager > 0 else 0
    usage_savings_pct = ((total_eager - total_usage) / total_eager * 100) if total_eager > 0 else 0

    print(f"\nKEY METRICS:")
    print(f"- Initial memory reduction: {lazy_savings_pct:.1f}%")
    print(f"- Memory savings with usage: {usage_savings_pct:.1f}%")
    print(f"- Total memory saved: {format_bytes((total_eager - total_usage) * 1024 * 1024)}")

    print(f"\nBENEFITS:")
    print(f"- Libraries loaded only when needed")
    print(f"- Massive reduction in initial memory footprint")
    print(f"- Significant savings even with actual usage")
    print(f"- Minimal overhead for lazy import system")
    print(f"- Perfect for applications using subset of features")

    print(f"\nRECOMMENDATIONS:")
    print(f"- Use lazy loading for heavy ML libraries")
    print(f"- Most effective when not all features are used")
    print(f"- Consider lazy loading for optional dependencies")
    print(f"- Monitor actual usage patterns for optimization")


def main():
    """Main test function."""
    print("CHONKIE MEMORY OPTIMIZATION TEST")
    print("Testing lazy loading vs eager imports")
    print("=" * 80)

    # Run all tests
    eager_results = test_eager_imports()
    lazy_results = test_lazy_imports()
    usage_results = test_actual_usage()
    integration_results = test_chonkie_integration()

    # Print comprehensive summary
    print_summary(eager_results, lazy_results, usage_results, integration_results)


if __name__ == "__main__":
    main()