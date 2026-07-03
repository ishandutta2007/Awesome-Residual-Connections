# Linear Identity Shortcuts

## Concept Diagram

```mermaid
flowchart LR
    X[Input x] -- Identity Shortcut --> Add((+))
    X --> F[Weight Block F] --> Add
    Add --> Out[Output y = F x + x]
```

## Detailed Information

The purest form of skip connection where the input is added directly to the output of the residual block without any parameterized projection or scaling: y = F(x) + x.

---
[Back to README](../README.md)
