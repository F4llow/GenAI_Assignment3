# Investigation of Pre-training vs. RAG for Java Bug Fixing
**Course:** CSCI 455/555 — GenAI for Software Development, William & Mary  
**Repository:** [https://github.com/F4llow/GenAI_Assignment3/](https://github.com/F4llow/GenAI_Assignment3/)

## Project Overview
This research project evaluates the impact of domain-specific pre-training on a 60M-parameter T5-Small architecture for automated Java bug fixing. We compare task-specific small models across two pipelines and contrast them against a Large Language Model (Qwen 1.5B) using Retrieval-Augmented Generation (RAG). All configurations are evaluated against the CodeXGLUE Code Refinement (medium) benchmark.

### Evaluated Configurations
| Configuration | Description |
| :--- | :--- |
| **Pipeline A** | T5-Small pre-trained on 50K Java methods (Span Corruption) → Fine-tuned on bug fixing. |
| **Pipeline B** | T5-Small initialized from scratch with random weights → Fine-tuned directly on bug fixing. |
| **Qwen Zero-Shot** | Qwen 2.5-Coder-1.5B-Instruct prompted with no examples. |
| **Qwen RAG (3-shot)** | Qwen 2.5-Coder-1.5B prompted with 3 retrieved bug-fix examples using a CodeBERT semantic retriever and FAISS index. |

---

## Key Findings & Comparative Analysis

1. **The Inference Efficiency Paradox** Despite Pipeline B (fine-tuned from scratch) achieving a higher CodeBLEU score, it suffered from a massive "babbling" penalty, resulting in a **10.5x increase in inference latency**. Pipeline A learned structural stopping cues (the `</s>` EOS token) during pre-training, allowing it to complete the test set evaluation in ~10 minutes. Pipeline B failed to learn these cues effectively, hallucinating code until reaching the hard token limit for nearly every sample, resulting in a ~1.7-hour evaluation time.

2. **The Initialization Trade-off** At this constrained corpus scale (50K methods, 3 epochs), fine-tuning from scratch (Pipeline B) yielded a significantly higher CodeBLEU (51.99) than the pre-trained variant (30.73). This suggests that for specialized sequence-to-sequence tasks with sufficient labeled data, direct convergence on the task objective may yield higher raw accuracy than forcing the model to "unlearn" a limited span-corruption pre-training objective.

3. **Tokenizer `<unk>` Collapse** During development, we identified and resolved a critical failure where legacy Hugging Face tokenizer wrappers caused the model to default to `<unk>` for valid Java code. The fix involved migrating to a Rust-based `PreTrainedTokenizerFast` backend with a Unigram SentencePiece model to properly parse whitespace.

4. **RAG Performance** Retrieval-Augmented Generation provided a noticeable boost to the Qwen model, increasing CodeBLEU from 42.84 to 44.51. This validates the use of CodeBERT for the semantic retrieval of analogous bug patterns rather than relying purely on zero-shot generalization.

---

## Results Summary

| Configuration | Exact Match | CodeBLEU | Runtime (Eval) |
| :--- | :--- | :--- | :--- |
| **Pipeline A (Pre-trained)** | 0.00% | 30.7260 | **9m 53s** |
| **Pipeline B (From Scratch)** | **0.0458%** | **51.9902** | 1h 43m 16s |
| **Qwen 1.5B Zero-Shot** | 0.00% | 42.8432 | 49m 23s |
| **Qwen 1.5B RAG (3-Shot)** | 0.00% | 44.5091 | 1h 00m 12s |

---

## Repository Structure

```text
.
├── callabresi_assignment3.ipynb   # Holistic notebook (Tokenizing, Pre-training, Fine-tuning, RAG)
├── report.pdf                     # Final 3-4 page research report
├── requirements.txt               # Project dependencies and version pins
├── final_results_summary.json     # Final benchmark scores for all paradigms
│
├── java_tokenizer/                # Saved Hugging Face Fast Tokenizer files
├── t5_final_pretrained/           # Final consolidated checkpoint of Pipeline A pre-training
├── t5_best_pipeline_A/            # Best fine-tuned model checkpoint for Pipeline A
├── t5_best_pipeline_B/            # Best fine-tuned model checkpoint for Pipeline B
├── t5_finetuned_pipeline_A/       # Training logs and checkpoints for Pipeline A
├── t5_finetuned_pipeline_B/       # Training logs and checkpoints for Pipeline B
├── t5_pretrained_checkpoints/     # Intermediate checkpoints from the pre-training phase
│
├── sp_code.model / .vocab         # Trained SentencePiece tokenizer (Unigram)
└── pretrain_corpus.txt            # Raw Java corpus (50K methods) used for tokenizer/pre-training
```

---

## Setup & Reproducibility

### Hardware Requirements
* **GPU:** An NVIDIA GPU with at least 16GB VRAM is highly recommended (a Quadro RTX 6000 was used for these results). Batched generation (BS=16) for Qwen 1.5B requires significant memory.
* **Architecture Note:** Turing architecture GPUs (like the RTX 6000) do not natively support `bf16`. The pre-training loop was stabilized using full `FP32` precision to avoid catastrophic gradient explosions.

### Installation
To reproduce the environment and install dependencies:

```bash
# 1. Create a virtual environment
python3.10 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```
*Note: `tree-sitter==0.20.4` is pinned specifically in the requirements to maintain compatibility with the `codebleu` library and avoid breaking API changes found in newer versions.*

### Running the Project
The entire project implementation is contained within the single, unified `callabresi_assignment3.ipynb` notebook. To reproduce the results, run the notebook cells in sequential order:

* **Cells 1-4:** Tokenizer training (SentencePiece Unigram) and CodeSearchNet data preparation.
* **Cells 5-7:** T5-Small manual initialization and Pre-training loop (Pipeline A).
* **Cells 8-10:** Fine-tuning both Pipeline A (from pre-trained weights) and Pipeline B (from scratch) on CodeXGLUE.
* **Cells 11-12:** T5 Evaluation (CodeBLEU and Exact Match calculation).
* **Cells 13-15:** CodeBERT Retriever setup, FAISS indexing, Qwen 1.5B batched Zero-Shot/RAG evaluation, and final JSON output.
