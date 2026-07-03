# The Unstructured Channel-Padding Core Stall

## Concept Diagram

```mermaid
flowchart TD
    Pad[Zero Padding / Strides] --> Stall[Tensor Core Alignment Stall]
    Fuse[Fused CUDA/Triton Kernel] --> Sync[Aligned Memory Access]
```

## Detailed Information

Unfused projection layers can cause memory stalls due to non-contiguous matrix layouts on Tensor Cores. Custom fused Triton or CUDA kernels optimize these operations into unified memory accesses.

---
[Back to README](../README.md)
