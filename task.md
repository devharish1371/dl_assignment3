In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. Transitioning from the convolutional neural networks used in previous assignments, you will now build a purely attention-based sequence-to-sequence model. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English.

- Base Paper: "Attention is all you need" https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
- Permitted Libraries: torch, numpy, matplotlib, scikit-learn, wandb, datasets, spacy, bleu, tqdm.
- Project Structure: Follow the official Assignment-3 GitHub Skeleton. https://github.com/MiRL-IITM/da6401_assignment_3.

To manage compute resources while achieving this, we will strictly limit the assignment to the following dataset:

**Multi30k Dataset:** A multilingual dataset designed specifically for training and evaluating Neural Machine Translation models in a resource-constrained environment - Comprises 29,000 training pairs, 1,014 validation pairs, and 1,000 test pairs (https://huggingface.co/datasets/bentrevett/multi30k).

**Note**
- You are expected to implement the base architecture from the paper.
- You can adopt any tokenization scheme available in the spacy library. All pre-processing should be done using this library only.
- Entire implementation should be in torch. You have to use basic building blocks in torch like nn.Linear and nn.Module to build the model and train it. You can implement any custom loss function or use existing loss functions in torch.
- For Layer Normalization use the nn.LayerNorm from torch.

---

**1 Implementation & Evaluation Requirements (50 Marks)**

**1.1 Task 1: Scaled Dot-Product and Multi-Head Attention**

Implement the Attention mechanism. You are not allowed to use torch.nn.MultiheadAttention.

- **Scaled Dot-Product Attention:** Implement the attention mechanism defined as:
  Attention(Q, K, V) = softmax(QKᵀ / √dk) V
- **Multi-Head Attention (MHA):** Implement the parallel attention heads that allow the model to jointly attend to information from different representation subspaces.
- **Masking:** Implement both padding masks (for encoder and decoder) and the look-ahead (causal) mask for the decoder to prevent positions from attending to subsequent positions.

---

**1.2 Task 2: Transformer Encoder and Decoder Stacks**

Construct the full encoder and decoder layers by following the exact sub-layer structure described in the paper.

- **Positional Encoding:** Implement the sinusoidal positional encoding to provide the model with information regarding the relative or absolute position of the tokens:
  PE(pos, 2i) = sin(pos / 10000^(2i/dmodel))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/dmodel))
- **Layer Normalization & Residuals:** Implement the "Add & Norm" structure. You may choose between Pre-LayerNorm or Post-LayerNorm, but you must justify your choice in the report.
- **Point-wise Feed-Forward Networks:** Implement the two-layer linear transformation with a ReLU activation in between:
  FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

---

**1.3 Task 3: Training Pipeline and Optimization**

To achieve convergence on the Multi30k dataset, you must implement specific optimization strategies mentioned in the original paper.

- **Label Smoothing:** Implement a label smoothing value of ε_ls = 0.1.
- **Noam Scheduler:** Implement the learning rate schedule with a warmup phase:
  lrate = d_model^(−0.5) · min(step_num^(−0.5), step_num · warmup_steps^(−1.5))
- **Greedy Decoding:** Write an inference function that generates translations token-by-token using the trained model.

**Automated Evaluation Pipeline**

The submission will be evaluated based on the following weighted criteria:

- **Multi-Head Attention [10M]:** Your scaled dot product attention and MultiHeadAttention will be tested across five criteria: correctness of output shape, attention weights summing to 1 over the key dimension, masked positions receiving zero attention weight, MultiHeadAttention output shape under varying d_model and num_heads, and causal masking producing different outputs from unmasked attention.
- **Positional Encoding [10M]:** Your PositionalEncoding will be tested across five criteria: output shape preservation, even-indexed dimensions equalling sin(0) = 0 at position 0, odd-indexed dimensions equalling cos(0) = 1 at position 0, formula correctness at an arbitrary (pos, dim) pair, and the encoding being registered as a buffer rather than a trainable parameter.
- **Noam LR Scheduler [10M]:** Your NoamScheduler will be tested across five criteria: learning rate being monotonically increasing during warm-up, the peak occurring within 10 steps of warmup_steps, learning rate being monotonically decreasing after warm-up, the peak value matching the closed-form formula, and the learning rate at step 1 matching the formula.
- **Test-Set Performance [20M]:** Your best saved checkpoint will be loaded and evaluated on a held-out test set using corpus-level BLEU score via evaluate_bleu.