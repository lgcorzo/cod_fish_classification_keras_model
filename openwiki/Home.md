---
iso_doc_type: "architecture_description"
iso_viewpoint: "context"
type: "wiki_home"
tags: ["faster-rcnn", "vgg16", "computer-vision", "cod-fish", "oleada-1"]
timestamp: "2026-11-22T17:52:45+01:00"
generated: true
verified: "false"
visibility: "public"
---

# `cod_fish_classification_keras_model` — Living Architecture Wiki

## System Context (C1)

```mermaid
flowchart LR
    Cam["Conveyor RGB/NIR Camera (10,942 captures)"] --> Pre["src/cod_fish_cv/config.py & Preprocessing"]
    Pre --> Model["Faster R-CNN (VGG-16 Backbone)"]
    Model --> Out["Species Classification (1_HYSE, 2_SEI, 3_TORSK, 4_LASK)"]
```

## Module Index
- `src/cod_fish_cv/config.py`: Canonical species classes, RPN anchor scales `(128, 256, 512)`, anchor ratios, and conveyor ROI bounds `(0.05, 0.95)`.
- `tests/test_config.py`: Golden-master configuration tests.
