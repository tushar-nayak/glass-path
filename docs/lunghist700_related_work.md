# LungHist700: Related Work Review (Classification)

Last updated: 2026-04-23

This note summarizes *publicly findable* studies that explicitly mention using the **LungHist700** dataset for classification, and compares them to the direction of this repo (`glass-path`).

## Dataset: what LungHist700 is

The dataset paper introduces LungHist700 as:
- 691 H&E histopathology images (1200x1600 JPGs)
- 45 patients (each image linked to a `patient_id` in a CSV)
- 20x and 40x magnification
- Labels: 3 superclasses (ACA, SCC, NOR) and 7 subclasses (differentiation levels for ACA/SCC + NOR)

Primary source: Diosdado et al., Scientific Data (2024), DOI `10.1038/s41597-024-03944-3`.

## What the ecosystem has done (by study)

The set below is not guaranteed exhaustive. It is skewed toward sources that are easy to verify from the open web.

| Study | Venue / Year | Task (labels) | Split / Protocol Notes | Model / Method | Key Takeaway |
|---|---|---|---|---|---|
| Diosdado et al. (dataset paper) | Scientific Data, 2024 | 3-class (ACA/NOR/SCC), discusses 7 subclasses | Reports results by magnification (20x vs 40x); provides baseline code | DNN baseline + MIL baseline | Positions LungHist700 as suitable for deep learning; provides baseline recipes and performance range |
| Kaveh et al., "VALF: Validation-Adaptive Focal Loss for Histopathology" | Eurasian Journal of Mathematical and Computer Applications, 2025 | 3-class (ACA/NOR/SCC) | Uses "official LungHist700 train/validation/test split" and follows dataset augmentation recipe; resizes to 300x400 | 5 CNN backbones (incl. ResNet50, EfficientNetV2B3, DenseNet121) + loss-function variants (CCE/WCCE/FL/VALF) | Claims consistent macro-F1 gains via a validation-adaptive reweighting focal loss across backbones/magnifications |
| Debnath et al., "Explainable Hybrid CNN–ViT Framework..." | Preprint (Authorea), 2026 | 3-class (ACA/NOR/SCC) | Reports a stratified split; typical resize/normalize/augment | ResNet50 + ViT-B/16 ensemble via weighted soft voting; Optuna tuning; Grad-CAM | Reports very high accuracy and adds explainability; preprint, not peer reviewed |
| "Cost-sensitive multi-kernel ELM..." (case study section) | PLOS One, 2024/2025 | Binary (normal vs unnormal) | Resizes to 300x400; 70/30 train/test; uses ResNet50 features as input to ELM; adds asymmetric misclassification costs | Cost-sensitive ELM classifier on top of ResNet features | Treats LungHist700 as a realistic cost-sensitive testbed; reframes to binary task |
| Putra (DenseNet121 LungHist700 classification analysis) | Undergraduate thesis (IPB University repository), 2025 | 3-class (ACA/NOR/SCC) | Not clearly comparable to official split from the abstract page alone | DenseNet121; explores CLAHE + hyperparameters | Reports strong accuracy/balanced accuracy with basic CNN training; thesis-level evidence, not peer reviewed |

## Patterns in prior work (what’s common)

1. Centralized supervised classification dominates.
   - Most work treats LungHist700 as a small image classification dataset: resize images to 224-ish or 300x400 and train ImageNet-style models (ResNet, DenseNet, EfficientNet, ViT).

2. MIL is recognized (by the dataset paper), but most later work still reports single-image classification.

3. Many papers do not clearly state patient-level splits.
   - This matters a lot for leakage risk. LungHist700’s distinguishing feature is that it ships **patient IDs**; work that ignores patient-level splitting can overestimate generalization.

4. Optimization knobs (losses, preprocessing like CLAHE, ensembling) are popular because the dataset is small.

## Where `glass-path` differs (and can be novel)

This repo’s core idea (as implemented) is:
- **Federated simulation**: patients are treated as clients (`patient_id` partitions)
- **Federated self-supervised pretraining** (BYOL-ish) followed by **federated supervised fine-tuning**

Modules exist for graph reasoning (`code/graph.py`) and concept bottlenecks (`code/concept.py`), but they are not yet wired into the main training command.

### Novel directions that are plausible on LungHist700

1. Federated SSL as the central contribution.
   - Compare:
     - Centralized supervised (strong baselines: ResNet50, ViT, DenseNet, EfficientNet)
     - Centralized SSL pretrain -> supervised fine-tune
     - Federated SSL pretrain -> federated supervised fine-tune (patient clients)
   - Primary claim: better patient-level generalization and a privacy-aligned training protocol.

2. Explicit patient-level evaluation protocol as a contribution.
   - Because patient IDs exist, publish a clear “no-leakage” protocol (patient-disjoint split) and re-evaluate baselines under it.
   - Even if method gains are modest, a *clean protocol* can be a meaningful contribution for small medical datasets where leakage is easy.

3. Magnification-aware learning (20x vs 40x).
   - Treat magnification as:
     - a domain shift problem (domain-adversarial / domain-invariant SSL),
     - a multi-task head (predict class + magnification),
     - or a paired-view problem (if any paired fields exist per patient/image_id).

4. Federated MIL.
   - Implement patch extraction + MIL pooling locally per client and aggregate federatedly.
   - Motivated by the dataset paper’s MIL baseline and by pathology practice (high-res images).

5. Graph + concept bottleneck integration (if you want interpretability beyond Grad-CAM).
   - Construct graphs from model tokens or from simple unsupervised structure (kNN on patch coordinates).
   - Use concept bottleneck as an interpretable intermediate, then quantify concept stability across magnification.

## Minimal “credible comparison” checklist (recommended)

To make comparisons to the above literature defensible:
- Use patient-disjoint splits (or reproduce the dataset’s official split exactly, if published).
- Report macro-F1 and balanced accuracy, not just accuracy.
- Report results separately for 20x and 40x, and combined.
- Avoid leakage from preprocessing/caching: do all augmentation inside the train split.

## References (links to verify)

- LungHist700 dataset paper (Scientific Data): DOI `10.1038/s41597-024-03944-3`
- VALF paper PDF (EJMCA): DOI `10.32523/2306-6172-2025-13-4-128-140`
- Hybrid CNN–ViT preprint (Authorea): DOI `10.22541/au.177187693.39332127/v1`
- PLOS One cost-sensitive ELM case study: DOI `10.1371/journal.pone.0314851`
- DenseNet121 thesis entry (IPB repository): handle `123456789/162553`

