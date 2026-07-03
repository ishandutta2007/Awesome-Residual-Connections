# The Plain Sequential Era

## Concept Diagram

```mermaid
flowchart TD
    In[Input] --> L1[Layer 1: Conv + Activation]
    L1 --> L2[Layer 2: Conv + Activation]
    L2 --> L3[Layer 3: Conv + Activation]
    L3 --> Out[Output]
    style In fill:#f9f,stroke:#333,stroke-width:2px
    style Out fill:#9f9,stroke:#333,stroke-width:2px
```

## Detailed Information

Before 2015, convolutional neural networks (such as AlexNet and VGG) were designed by stacking layer upon layer sequentially. As networks grew deeper, they suffered from the vanishing/exploding gradient problem, causing training degradation where accuracy saturated and then declined rapidly.

---
[Back to README](../README.md)
