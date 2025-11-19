# LLM Batching Optimization

This notebook demonstrates **batching techniques** for Large Language Model (LLM) inference, focusing on how batching affects **throughput** and **latency** in text generation.

## What is Batching?

**Batching** is the technique of processing multiple requests simultaneously instead of one at a time. Think of it like:
- **Batching**: Loading multiple passengers on a bus (efficient)
- **No Batching**: Each passenger takes their own car (inefficient)

## Key Concepts Covered

### 1. **Padding & Attention Masks**
- **Problem**: Different prompts have different lengths
- **Solution**: Pad shorter sequences to match the longest one
- **Attention Mask**: Tells the model which tokens are real vs padding

# Example with 3 prompts of different lengths
prompts = [
    'The quick brown fox jumped over the',  # 7 tokens
    'Never have I ever',                    # 4 tokens  
    'What comes up must'                    # 4 tokens
]

# After padding (left-padded):
# [1, 1, 1, 1, 1, 1, 1]  ← all real tokens
# [0, 0, 0, 1, 1, 1, 1]  ← 3 padding + 4 real
# [0, 0, 0, 1, 1, 1, 1]  ← 3 padding + 4 real### 2. **Position IDs**
- **Purpose**: Tell the model the position of each token in its sequence
- **Generation**: `position_ids = attention_mask.cumsum(dim=-1) - 1`
- **Result**: Padding tokens get position -1, real tokens get 0, 1, 2, 3...

### 3. **KV Caching with Batching**
- Reuses previous computations for faster generation
- Works seamlessly with batched inputs
- Significantly reduces computational overhead

## Performance Analysis

The notebook measures two key metrics:

### **Throughput** 
- **Definition**: Total tokens generated per second
- **Formula**: `(batch_size × max_tokens) / total_time`
- **Higher is better**

### **Latency**
- **Definition**: Average time to generate each token
- **Formula**: `total_time / max_tokens`  
- **Lower is better**

## Key Findings

The experiments test batch sizes from 1 to 128 and reveal:

1. **Throughput increases dramatically** with larger batch sizes
   - Batch size 1: ~34 tokens/sec
   - Batch size 128: ~1,482 tokens/sec
   - **~44x improvement!**

2. **Latency initially decreases, then increases**
   - Sweet spot around batch size 4-8
   - Beyond that, memory/compute constraints kick in

3. **Optimal batch size** balances throughput and latency based on your use case

## Technical Implementation

### Core Functions

1. **`generate_token_kv_caching()`**: Generates single token with KV caching
2. **`generate_batch()`**: Handles batched text generation with proper:
   - Position ID management
   - Attention mask extension  
   - KV cache handling

### Key Technical Challenges Solved

- **Dynamic attention masks**: Extending masks as new tokens are generated
- **Position ID calculation**: Proper positioning for padded sequences
- **Batch dimension consistency**: Ensuring tensor shapes match across operations

## Visualization

The notebook includes plots showing:
- **Throughput vs Batch Size**: Exponential improvement curve
- **Latency vs Batch Size**: U-shaped curve with optimal point

## Learning Outcomes

After working through this notebook, you'll understand:

1. **Why batching matters** for LLM inference efficiency
2. **How to implement batching** with proper padding and attention masks
3. **Performance trade-offs** between throughput and latency
4. **Technical challenges** in batched text generation
5. **How to measure and optimize** LLM inference performance

## Usage

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("../models/gpt2")
tokenizer = AutoTokenizer.from_pretrained("../models/gpt2")

# Prepare batched inputs
prompts = ["Your prompt 1", "Your prompt 2", "Your prompt 3"]
inputs = tokenizer(prompts, padding=True, return_tensors="pt")

# Generate with batching
generated_tokens = generate_batch(inputs, max_tokens=10)## Related

- **Text Generation with KV Caching**: Foundation techniques used here
- **Attention Mechanisms**: Understanding how attention works with padding

---

*This experiment demonstrates the power of batching in making LLM inference more efficient - a crucial technique for production deployments!*