# Symmetrical Lateral Skip Connections (U-Net)

## Concept Diagram

```mermaid
flowchart LR
    E1[Encoder Layer 1] -- Symmetrical Skip --> D1[Decoder Layer 1]
    E1 --> E2[Encoder Layer 2] --> D1
```

## Detailed Information

Common in image segmentation (e.g., U-Net), these skip connections copy high-resolution spatial feature maps from the encoder side directly to the decoder side, preserving fine details that would otherwise be lost during downsampling.

---
[Back to README](../README.md)
