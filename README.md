# melikechan's [TorchLeet](https://github.com/Exorust/TorchLeet) solutions

> [!WARNING]
> This repository only contains my solutions, which are _usually_ identical to the solutions in the [original repository](https://github.com/Exorust/TorchLeet).

## Table of Contents

- [Question Set](#question-set)
  - [🔵Basic](#basic)
  - [🟢Easy](#easy)
  - [🟡Medium](#medium)
  - [🔴Hard](#hard)
- [LLM Set](#llm-set)

## Question Set

### 🔵Basic

1. [Implement linear regression](torch/basic/linear-regression.ipynb) ✅
2. [Write a custom Dataset and Dataloader to load from a CSV file](torch/basic/custom-dataset/custom-dataset.ipynb) ✅
3. [Write a custom activation function (Simple)](torch/basic/custom-activation.ipynb) ✅
4. [Implement Custom Loss Function (Huber Loss)](torch/basic/custom-loss.ipynb) ✅
5. [Implement a Deep Neural Network](torch/basic/custom-dnn.ipynb) ✅
6. [Visualize Training Progress with TensorBoard in PyTorch](torch/basic/tensorboard.ipynb)
7. [Save and Load Your PyTorch Model](torch/basic/save-model/save-model.ipynb) ✅
8. [Implement Softmax function from scratch](torch/basic/softmax.ipynb)

---

### 🟢Easy

1. [Implement a CNN on CIFAR-10](torch/easy/cnn.ipynb)
2. [Implement an RNN from Scratch](torch/easy/rnn.ipynb)
3. [Use `torchvision.transforms` to apply data augmentation](torch/easy/augmentation.ipynb)
4. [Add a benchmark to your PyTorch code](torch/easy/benchmark.ipynb)
5. [Train an autoencoder for anomaly detection](torch/easy/autoencoder.ipynb)
6. [Quantize your language model](torch/easy/quantize-language-model.ipynb)
7. [Implement Mixed Precision Training using torch.cuda.amp](torch/easy/cuda-amp.ipynb)

---

### 🟡Medium

1. [Implement parameter initialization for a CNN](torch/medium/cnn-param-init.ipynb)
2. [Implement a CNN from Scratch](torch/medium/cnn-from-scratch.ipynb)
3. [Implement an LSTM from Scratch](torch/medium/lstm.ipynb)
4. [Implement AlexNet from scratch](torch/medium/alexnet.ipynb)
5. Build a Dense Retrieval System using PyTorch
6. Implement KNN from scratch in PyTorch
7. [Train a 3D CNN network for segmenting CT images](torch/medium/3dcnn.ipynb)

---

### 🔴Hard

1. [Write a custom Autograd function for activation (SILU)](torch/hard/custom-autograd-function.ipynb)
2. Write a Neural Style Transfer  
3. Build a Graph Neural Network (GNN) from scratch
4. Build a Graph Convolutional Network (GCN) from scratch
5. [Write a Transformer](torch/hard/transformer.ipynb)
6. [Write a GAN](torch/hard/gan.ipynb)
7. [Write Sequence-to-Sequence with Attention](torch/hard/seq2seq-with-attention.ipynb)
8. [Enable distributed training in pytorch (DistributedDataParallel)]
9. [Work with Sparse Tensors]
10. [Add GradCam/SHAP to explain the model.](torch/hard/xai.ipynb)
11. Linear Probe on CLIP Features
12. Add Cross Modal Embedding Visualization to CLIP (t-SNE/UMAP)
13. Implement a Vision Transformer
14. Implement a Variational Autoencoder

---

## LLM Set

1. Implement KL Divergence Loss
2. [Implement RMS Norm](llm/rms-norm.ipynb)
3. [Implement Byte Pair Encoding from Scratch](llm/bpe.ipynb)
4. Create a RAG Search of Embeddings from a set of Reviews
5. Implement Predictive Prefill with Speculative Decoding
6. [Implement Attention from Scratch](llm/attention.ipynb)
7. [Implement Multi-Head Attention from Scratch](llm/multi-head-attention.ipynb)
8. [Implement Grouped Query Attention from Scratch](llm/grouped-query-attention.ipynb)
9. Implement KV Cache in Multi-Head Attention from Scratch
10. [Implement Sinusoidal Embeddings](llm/sinusoidal.ipynb)
11. [Implement ROPE Embeddings](llm/rope.ipynb)
12. [Implement SmolLM from Scratch](llm/smollm.ipynb)
13. Implement Quantization of Models
    - GPTQ
14. Implement Beam Search atop LLM for decoding
15. Implement Top K Sampling atop LLM for decoding
16. Implement Top p Sampling atop LLM for decoding
17. Implement Temperature Sampling atop LLM for decoding
18. Implement LoRA on a layer of an LLM
    - QLoRA
19. Mix two models to create a mixture of Experts
20. Apply SFT on SmolLM
21. Apply RLHF on SmolLM
22. Implement DPO based RLHF
23. Add continuous batching to your LLM
24. Chunk Textual Data for Dense Passage Retrieval
25. Implement Large scale Training => 5D Parallelism

## Contribution, Authors, etc

See the [original repository](https://github.com/Exorust/TorchLeet).
