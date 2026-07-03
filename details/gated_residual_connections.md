# Gated Residual Connections

## Concept Diagram

```mermaid
flowchart TD
    X[Input x] --> T[Gating Function T x]
    X --> F[Weight Block F]
    F --> Mult1((x))
    T --> Mult1
    X --> Mult2((x))
    T -- "1 - T(x)" --> Mult2
    Mult1 --> Add((+))
    Mult2 --> Add
    Add --> Out[Output y]
```

## Detailed Information

In Highway Networks (2015), the flow of information through the shortcut is dynamically controlled using learnable gating units: y = F(x) * T(x) + x * (1 - T(x)).

---
[Back to README](../README.md)
