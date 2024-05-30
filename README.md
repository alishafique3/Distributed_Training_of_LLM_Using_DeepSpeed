# Distributed training of LLM using deepspeed for text classification task
Training or fine-tuning Large Language Models (LLMs) involves dealing with incredibly large models and datasets. These models can have billions of parameters and require vast amounts of GPU memory to train. Not only do the model weights take up a lot of memory, but the optimizer states also add to the memory requirements. With traditional methods, storing copies of the model weights, momentum, and variance parameters can quickly consume GPU memory. To overcome these challenges, distributed training strategies are employed. Distributed training allows for parallel processing across multiple devices, reducing memory usage and speeding up training. It's a crucial technique for efficiently training LLMs and advancing natural language understanding and generation.

## Usage:
The code is built using the NVIDIA container image of Pytorch, release 23.10, which is available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). It includes:

- Ubuntu 22.04 including Python 3.10
- NVIDIA cuDNN 8.9.5
- PyTorch 23.10

Also install the libraries inside the docker
```
bash install.sh
```
## Baseline Model - DistilBert
Text classification example is used as base code which is available at [Link](https://huggingface.co/docs/transformers/en/tasks/sequence_classification), accessed on April 16, 2024. This [blog](https://sumanthrh.com/post/distributed-and-efficient-finetuning/#more-on-deepspeed-and-fsdp) is also helpful for this tutorial. The model used for this tutorial is DistilBERT which is an encoder-based transformers model, smaller and faster than BERT. It was pretrained on the same corpus in a self-supervised fashion, using the BERT base model as a teacher.

The deepspeed library is utilized to implement the ZeRO Stage 2 algorithm for fine-tuning the model. Distributed training is used across two GPUs (Tesla P4). I have specified the DeepSpeed configuration file path as "/<path_to_config_file>/ds_config_zero2.json" in the training arguments. Additionally, I have set the ```report_to``` argument to 'wandb' to report the results to the Weights and Biases dashboard. It's important to note that, for training, the configurations between DeepSpeed and Trainer arguments can be overlapping. In such cases, DeepSpeed configurations take precedence over Trainer settings. To prevent any potential conflicts and errors, training settings can be defined using Trainer arguments, while DeepSpeed settings are set to "auto".  ```torchrun``` or ```deepspeed``` launchers can be used to launch the code for distributed training.

```bash
deepspeed --num_gpus 2 sequence_classification.py --deepspeed_config /<path_to_config_file>/ds_config_zero2.json
```
Three python files are also provided in basics folder in order to understand the concepts of distributed learning from scratch using pytorch libraries. This code is taken from following [github](https://github.com/seba-1511/dist_tuto.pth/tree/gh-pages) and [link](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html).

## Distributed Training Concepts
Collective operations, in the context of parallel computing, refer to communication patterns where multiple processes or threads collaborate to perform a single operation collectively. These operations typically involve exchanging data among processes in a coordinated manner to achieve a common goal.

Common collective operations include:
- **Broadcast**: One process sends the same data to all other processes.
- **Reduce**: All processes combine their data using an operation (such as sum, min, max) and store the result in one process.
- **All-gather**: All processes gather data from all other processes and store the combined data locally.
- **All-reduce**: All processes combine their data using an operation (such as sum, min, max) and store the result in all processes.
- **Scatter**: One process distributes data to all other processes, with each process receiving a subset of the data.
- **Gather**: All processes send their data to one process, which collects and combines the data.
These collective operations are fundamental for distributed computing frameworks like NCCL (NVIDIA Collective Communications Library) and MPI (Message Passing Interface). These are used extensively in parallel algorithms and applications. They enable efficient coordination and communication among distributed processes, leading to improved scalability and performance. The following image has been adapted from [Link](https://pytorch.org/tutorials/intermediate/dist_tuto.html#setup).

![collective](https://github.com/alishafique3/Distributed_Training_of_LLM_Using_DeepSpeed/assets/17300597/94c6df80-ec17-40d2-afea-a5d180988c73)

Different communication frameworks are used in distributed computing are GLOO, MPI, and NCCL, particularly for deep learning training across multiple CPUs, GPUs, or machines.
- **GLOO**: GLOO is a collective communication library developed by Facebook. It is designed to support efficient communication primitives such as broadcast, all-gather, reduce, etc., necessary for distributed training.
- **MPI** (Message Passing Interface): MPI is a standardized and widely used communication library for parallel computing. It provides a set of functions for point-to-point and collective communication among processes running on different nodes in a distributed system.
- **NCCL** (NVIDIA Collective Communications Library): NCCL is a communication library developed by NVIDIA specifically for GPU-based parallel computing. It's optimized for communication between GPUs within a single node or across multiple nodes and provides high-performance primitives for collective operations like all-reduce, all-gather, etc.

`torch.distributed` supports these three frameworks as backends, each with different capabilities. You can see the table depicting various collective functions supported by each communication framework using `torch.distributed` library [Link](https://pytorch.org/docs/stable/distributed.html#backends).



## Distributed Training Strategies:
Using various communication libraries, the following distributed training strategies are developed:

![parallelism](https://github.com/alishafique3/Distributed_Training_of_LLM_Using_DeepSpeed/assets/17300597/70c22512-af3f-4205-85d4-0527fbb59eb9)

- **Data Parallelism (DP)**: In distributed training, each GPU worker handles a portion of the data and calculates the gradients based on that data. Afterward, all the gradients are combined and averaged across all workers to update the model weights. In PyTorch's Distributed Data Parallel (DDP), each GPU stores its copy of the model, optimizer, and gradients for its part of the data. Even with just two GPUs, users can see faster training thanks to PyTorch's built-in features like Data Parallel (DP) and Distributed Data Parallel (DDP). It is recommended to use DDP as it's more reliable and works with all models, whereas DP might not work with some models.
  
- **Tensor Parallelism (TP)**: In tensor parallelism, every GPU handles just a piece of a large tensor by slicing it horizontally across all the GPUs. Each GPU works on the same batch of data but focuses only on its portion of the model's weights. They share the parts of the model that each needs and compute the activations and gradients accordingly. Essentially, it's like dividing a big task into smaller chunks and having each GPU work on its slice to put everything together
  
- **Model Parallelism/ Vertical Model Parallelism (MP)**: In model parallelism, models are sliced vertically, with different layers of the model placed on different GPU workers.
  
- **Pipeline Parallelism (PP)**: In naive model parallelism, all GPUs process the same batch of data but wait for the previous GPU to finish its computation before proceeding. Essentially, only one GPU is active at any given time, leaving the others idle. This approach, though straightforward, isn't very efficient. A step up is Pipeline Parallelism (PP), where computation for different micro-batches of data overlaps, creating the illusion of parallelism. It's akin to the classic pipeline structure in computer architecture, where tasks are divided and processed simultaneously, optimizing efficiency.

![model_vs_pipeline](https://github.com/alishafique3/Distributed_Training_of_LLM_Using_DeepSpeed/assets/17300597/84f59fc0-c81c-45bd-9752-3a71c580add4)

To accommodate the model within memory constraints, current solutions make trade-offs between computation, communication, and development efficiency:

- Data parallelism fails to reduce the memory usage per device for models exceeding 1 billion parameters, even on GPUs with 32GB capacity.
- Model parallelism struggles to scale efficiently beyond a single node due to complex computation and costly communication. Implementing model parallelism frameworks often demands substantial code integration, which may be specific to model architecture.

ZeRO, developed by Microsoft, aims to overcome the constraints of both data parallelism and model parallelism while retaining their respective advantages.

## ZeRO - Data parallelism and Tensor Parallelism: [paper](https://arxiv.org/abs/1910.02054v3)
ZeRO (Zero Redundancy Optimizer) tackles memory redundancies in data-parallel processes by distributing the model states—parameters, gradients, and optimizer state—across these processes instead of duplicating them. It employs a dynamic communication schedule during training to exchange the required state among distributed devices, preserving the computational granularity and communication volume essential for data parallelism.

This is one of the most efficient and popular strategies for distributed training at the moment. DeepSpeed’s ZeRO, or Zero Redundancy Optimizer, is a form of data parallelism and tensor parallelism that massively improves memory efficiency. DeepSpeed ZeRO includes all the ZeRO stages 1, 2, and 3 as well as ZeRO-Offload, and ZeRO-Infinity (which can offload to disk/NVMe). ZeRO++. This algorithm can be visualized in the following diagram taken from this [blog post](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/):

![DeepSpeed-Image-1](https://github.com/alishafique3/Distributed_Training_of_LLM_Using_DeepSpeed/assets/17300597/58c1d527-fa10-4843-8280-cbc018dad90a)

This figure shows the memory savings and communication volume for the three stages of ZeRO compared with standard data parallel baseline. In the memory consumption formula, Ψ refers to the number of parameters in a model and K is the optimizer specific constant term. As a specific example, we show the memory consumption for a 7.5B parameter model using Adam optimizer where K=12 on 64 GPUs.


## Understanding of ZeRO with example
Understanding this concept may seem tricky at first, but it's pretty straightforward. It's just like regular DataParallel (DP), but with a twist. Instead of each GPU holding a copy of the entire model, gradients, and optimizer states, they only store a portion of it. This is called horizontal model slicing. When the full layer parameters are needed during runtime, all GPUs communicate together to share the parts they're missing, so each GPU has what it needs to complete the task.
To understand the ZeRO algorithm, an example is taken from [Huggingface blog](https://huggingface.co/docs/transformers/v4.23.1/en/perf_train_gpu_many#zero-data-parallelism). Consider this simple model with 3 layers, where each layer has 3 params:

La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2

Layer La has weights a0, a1, and a2.

If we have 3 GPUs, the Sharded DDP (= Zero-DP) splits the model onto 3 GPUs like so:

GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2

In a way, this is the same horizontal slicing, as tensor parallelism, if you imagine the typical DNN diagram. Vertical slicing is where one puts whole layer-groups on different GPUs. But it’s just the starting point.

Now each of these GPUs will get the usual mini-batch as it works in DP:
x0 => GPU0
x1 => GPU1
x2 => GPU2
The inputs are unmodified - they think they are going to be processed by the normal model.

First, the inputs hit layer La.

Let’s focus just on GPU0: x0 needs a0, a1, and a2 params to do its forward path, but GPU0 has only a0 - it gets sent a1 from GPU1 and a2 from GPU2, bringing all pieces of the model together.

In parallel, GPU1 gets mini-batch x1 and it only has a1, but needs a0 and a2 params, so it gets those from GPU0 and GPU2.

The same happens to GPU2 which gets input x2. It gets a0 and a1 from GPU0 and GPU1, and with its a2 it reconstructs the full tensor.

All 3 GPUs get the full tensors reconstructed and a forward happens.

As soon as the calculation is done, the data that is no longer needed gets dropped - it’s only used during the calculation. The reconstruction is done efficiently via a pre-fetch.

The whole process is repeated for layer Lb, then Lc forward-wise, and then backward Lc -> Lb -> La.

ZeRO Stage 1, 2 and 3 animation can be seen from this [blog post](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

### ZeRO Stage 1: 
Shards optimizer states across GPUs

### ZeRO Stage 2: 
Shards optimizer states + gradients across GPUs

### ZeRO Stage 3: 
Shards optimizer states + gradients + model parameters GPUs. The diagram is taken from this [blog post](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/):

![ZeRO3](https://github.com/alishafique3/Distributed_Training_of_LLM_Using_DeepSpeed/assets/17300597/ab4d590f-84c6-4173-8971-578d36aa4813)

### ZeRO-Offload: [paper](https://arxiv.org/abs/2101.06840)
ZeRO-Offload is a smart way to train large models more efficiently. Released in January 2021, it lets the host CPU take on some of the work from the GPUs, like handling optimization tasks. This frees up GPU power for other important tasks. Offloading work to the CPU is slower than using the GPU, but ZeRO-Offload is smart about it. It only offloads less intensive tasks to the CPU, keeping the overall complexity the same. This means things like norm calculations and weight updates can happen on the CPU, while the GPU handles the heavy lifting like matrix multiplication during the forward and backward passes. ZeRO-Offload is compatible with all stages of ZeRO (1, 2, and 3), making it a versatile tool for efficient model training.

### ZeRO-Infinity: [paper](https://arxiv.org/abs/2104.07857)
It is an improvement over ZeRO-Offload which came up in April 2021, by allowing offloading to disk (NVMe memory), and making some improvements to CPU offloading. ZeRO-Infinity is specifically built on top of ZeRO-3. In their evaluations of model speed on 512 GPUs across 32 DGX-2 nodes, the authors showed that ZeRO-Infinity trains up to 20 trillion parameter models with throughput of up to 49 TFlops/GPU. There are some bandwidth requirements for ZeRO-Infinity such as NVMe-CPU and CPU-GPU communication.

### ZeRO++: [paper](https://arxiv.org/abs/2306.10209)
ZeRO++ is an enhanced version of ZeRO (Zero Redundancy Optimizer) developed by the DeepSpeed team. It introduces several key improvements aimed at optimizing memory usage and communication during distributed training of deep learning models. These enhancements include features like quantized weights (qwZ), hierarchical partitioning (hpZ), and quantized gradients (qgZ), which collectively aim to reduce communication volume, improve scalability, and enhance overall training efficiency.

- Quantized Weights (qwZ): This feature reduces parameter communication volume during all-gather operations by the quantization of model weights to int8.
- Hierarchical Partitioning (hpZ): Introducing a hybrid partitioning scheme, hierarchical partitioning facilitates multi-node settings with DeepSpeed ZeRO 3. It allows model parameter sharding within a node while replicating across nodes. It mitigates expensive inter-node parameter communication overhead, thereby enhancing overall throughput.
- Quantized Gradients (qgZ): This feature enables further reductions in communication volume by substituting fp16 with int4 quantized data during gradient reduce-scatter operations. 

Overall, ZeRO++ reduces communication volume by 4x with these three improvements, compared to ZeRO-3.

This image is taken from this YouTube video: [Microsoft DeepSpeed introduction at KAUST](https://www.youtube.com/watch?v=wbG2ZEDPIyw&t=2651s)

![GetImage(7)](https://github.com/alishafique3/Distributed_Training_of_LLM_Using_DeepSpeed/assets/17300597/1ff90688-8e4f-482c-904f-90e150df6c7e)

## Conclusion
In this tutorial, we studied various distributed training concepts and strategies. We also studied different variants of the Zero Redundancy Optimizer technique for efficiently training or finetuning large language models over multiple devices. Distributed training with ZeRO technique, allows us for parallel processing across multiple devices, reducing memory usage and speeding up training. It's an important and recent technique for efficiently training LLMs and advancing natural language processing.

## References
1.  Everything about Distributed Training and Efficient Finetuning Link: https://sumanthrh.com/post/distributed-and-efficient-finetuning/
2.  Efficient Training on Multiple GPUs Link: https://huggingface.co/docs/transformers/v4.23.1/en/perf_train_gpu_many#efficient-training-on-multiple-gpus
3.  ZeRO & DeepSpeed: New system optimizations enable training models with over 100 billion parameters Link: https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/
