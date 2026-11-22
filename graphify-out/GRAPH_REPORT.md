# Graph & AST Comprehension Report: `lgcorzo/cod_fish_classification_keras_model`

**Generated**: 2026-11-22T17:49:10+01:00  
**Wave**: Oleada 1 (Tier 3 Pilot)  
**CRG Metrics**: 4 notebooks parsed, 98 nodes, 1,777 edges  

## 1. Legacy Entrypoints & Artifacts
- `Object_Detection_DataPreprocessing_granit.ipynb`: Conveyor RGB/NIR image stitching, bounding-box coordinate scaling, and `annotation.txt` / `test_annotation.txt` generation.
- `frcnn_train_vgg.ipynb`: Faster R-CNN training with VGG-16 backbone, RPN anchor scales/ratios, RoI pooling, and `model_vgg_config.pickle` / `model/record.csv` output.
- `frcnn_test_vgg.ipynb`: Standard Faster R-CNN inference and species classification (`1_HYSE`, `2_SEI`, `3_TORSK`, `4_LASK`).
- `frcnn_test_vgg_granit.ipynb`: Granit dataset evaluation and morphometric length estimation.

## 2. Structural Gaps vs. Dark Gravity Standard
- Missing modular `src/cod_fish_cv/` package structure (all logic embedded in `.ipynb` notebooks).
- Missing `pyproject.toml`, `tests/` golden-master suite, `.specify/`, `specs/`, `.agents/skills/`, `openwiki/`, `evidence/latest/`, `AGENTS.md`, and `SECURITY.md`.
