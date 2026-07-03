# Pre-LayerNorm Transformer Scaling Era

## Concept Diagram

```mermaid
flowchart TD
    X[Input x] --> LN[Layer Norm]
    LN --> Sub[Sub-Layer: Attention/MLP]
    Sub --> Add((+))
    X --> Add
    Add --> Out[Output y]
```

## Detailed Information

Modern Large Language Models standardize on the Pre-LayerNorm (Pre-LN) configuration. Normalization is applied at the input of each sub-layer (attention and feed-forward networks), preserving the main identity path completely un-normalized for stable training scale-invariance.

---
[Back to README](../README.md)
