# Dense Cross-Layer Connections

## Concept Diagram

```mermaid
flowchart TD
    X0[Layer 0] --> X1[Layer 1: Concat x0]
    X0 --> X2[Layer 2: Concat x0, x1]
    X1 --> X2
    X2 --> Out[Output]
```

## Detailed Information

DenseNet replaces summation with concatenation. Each layer obtains inputs from all preceding layers and passes its own feature maps to all subsequent layers: y = [x_0, x_1, x_2, ...].

---
[Back to README](../README.md)
