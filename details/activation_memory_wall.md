# The Activation Memory Wall

## Concept Diagram

```mermaid
flowchart TD
    Fwd[Forward Pass: Save Checkpoints Only] --> Remat[Recompute Activations in Backward] --> Grad[Calculate Gradients]
```

## Detailed Information

Training deep models requires substantial GPU VRAM to store intermediate activations for backpropagation. Gradient checkpointing mitigates this by keeping only specific check-pointed activations and recomputing the rest during the backward pass.

---
[Back to README](../README.md)
