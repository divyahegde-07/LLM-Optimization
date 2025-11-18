# Text Generation with KV Caching

This project demonstrates how **Key-Value (KV) Caching** can significantly improve the performance of autoregressive text generation models by reducing token generation latency.

## Project Overview

KV Caching is an optimization technique that stores intermediate attention tensors during text generation, eliminating the need to recompute them for each new token. This project uses GPT-2 to compare generation performance with and without KV caching.

## Key Results

- **Without KV Caching**: 0.272 seconds for 10 tokens
- **With KV Caching**: 0.189 seconds for 10 tokens
- **Performance Improvement**: ~30% faster generation

## Features

- **GPT-2 Model Implementation**: Uses HuggingFace's transformers library
- **Performance Comparison**: Side-by-side analysis of caching vs non-caching approaches
- **Visualization**: Plots showing generation time per token
- **Local Model Storage**: Saves model and tokenizer for offline use

