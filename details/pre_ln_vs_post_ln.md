# Pre-LN vs. Post-LN Transformer Paths

## Concept Diagram

```mermaid
flowchart TD
    subgraph Post-LN
        X1[Input] --> Sub1[Sub-Layer] --> Add1((+))
        X1 --> Add1 --> LN1[Layer Norm] --> Out1[Output]
    end
    subgraph Pre-LN
        X2[Input] --> LN2[Layer Norm] --> Sub2[Sub-Layer] --> Add2((+))
        X2 --> Add2 --> Out2[Output]
    end
```

## Detailed Information

Post-LN places normalization after the residual addition, which requires warmup schedules because gradient norms near output layers are significantly larger. Pre-LN places normalization before the sub-layers, stabilizing training and eliminating warmup needs.

---
[Back to README](../README.md)
