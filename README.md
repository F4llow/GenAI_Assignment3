Investigation of Pre-training vs. RAG for Java Bug Fixing

CSCI 455/555 — GenAI for Software Development William & Mary Name: [Your Name]

Project Overview

This research project evaluates the impact of domain-specific pre-training on a 60M-parameter T5-Small architecture for automated Java bug fixing. We compare four distinct configurations against the CodeXGLUE Code Refinement (medium) benchmark:

Configuration

Description

Pipeline A

T5-Small pre-trained on 50K Java methods (Span Corruption) → Fine-tuned on bug fixing.

Pipeline B

T5-Small initialized from scratch → Fine-tuned directly on bug fixing.

Qwen Zero-Shot

Qwen 2.5-Coder-1.5B-Instruct prompted with no examples.

Qwen RAG (3-shot)

Qwen 2.5-Coder-1.5B with a CodeBERT semantic retriever and FAISS index.

Key Findings

1. The Inference Efficiency Discovery

While Pipeline B (from scratch) achieved a higher CodeBLEU score, it suffered from a 10.5x increase in inference latency. Pipeline A (pre-trained) learned structural stopping cues (the </s> token) during pre-training, allowing it to complete evaluation in ~10 minutes. Pipeline B failed to learn these cues effectively, hallucinating code until reaching sequence limits, resulting in a ~1.7 hour evaluation time.

2. Tokenizer  Collapse

During development, we identified and resolved a critical failure where legacy Hugging Face tokenizer wrappers caused the model to default to <unk> for valid Java code. The fix involved migrating to a Rust-based PreTrainedTokenizerFast backend with a Unigram SentencePiece model.

Repository Structure

.
├── callabresi_assignment3.ipynb   # Holistic Notebook (Tokenizing, Pre-training, Fine-tuning, RAG)
├── report.pdf                     # Final 3-4 page research report
├── requirements.txt               # Dependency list
├── final_results_summary.json     # Final metric outputs
├── sp_code.model / .vocab        # Trained SentencePiece tokenizer
└── java_tokenizer/                # Tokenizer config and metadata


Setup & Reproducibility

Hardware Requirements

GPU: NVIDIA Quadro RTX 6000 (or equivalent 24GB VRAM GPU) is recommended.

VRAM: Batched generation (BS=16) for Qwen 1.5B requires significant memory; memory cleanup is implemented in the notebook.

Installation

# Create environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt


Running the Project

The project is contained within a single holistic Jupyter Notebook.

Cell 1-4: Tokenizer training and data preparation.

Cell 5-7: T5-Small initialization and Pre-training (Pipeline A).

Cell 8-10: Fine-tuning both Pipeline A and Pipeline B.

Cell 11-12: T5 Evaluation (CodeBLEU/Exact Match).

Cell 13-15: CodeBERT Retriever setup, FAISS indexing, and Qwen 1.5B RAG Evaluation.

Results Summary

Configuration

Exact Match

CodeBLEU

Runtime (Eval)

Pipeline A (Pre-trained)

0.00%

30.726

9m 53s

Pipeline B (From Scratch)

0.0458%

51.9902

1h 43m 16s

Qwen 1.5B Zero-Shot

0.00%

42.8432

49m 23s

Qwen 1.5B RAG (3-Shot)

0.00%

44.5091

1h 00m 12s
