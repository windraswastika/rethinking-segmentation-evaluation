# Abstract

**Rethinking Segmentation Evaluation in Breast Ultrasound: A Systematic Analysis of Metric Sensitivity to Negative Cases**

---

## Background

The Dice coefficient is the de facto evaluation metric for medical image segmentation, yet its behavior on empty ground-truth masks — corresponding to normal (lesion-free) cases — remains underexamined. When negative cases are included in a benchmark dataset, the metric collapses to zero for any prediction containing even a single false-positive pixel, regardless of error magnitude. This binary penalty produces a sharp discontinuity that distorts mean performance scores and, critically, can invert model rankings — leading to reproducibility failures in comparative studies.

## Methods

We present a systematic four-phase analysis of this phenomenon on the BUSI dataset (780 breast ultrasound images; 647 positive, 133 normal). Phase 1 formalizes the mathematical pathology of Dice on empty masks and characterizes inconsistent behavior across six widely-used implementations (MONAI, segmentation-models-pytorch, torchmetrics, scikit-learn, and two custom variants). Phase 2 quantifies metric distortion through bootstrap simulation (200 runs, seed = 42) across normal-case proportions ranging from 0% to 50% and false-positive rates from 0.1 to 0.8. Phase 3 trains five architectures (U-Net, Attention U-Net, U-Net++, MA-Net, LinkNet; all with ResNet-34 encoder) under identical conditions and evaluates them using three evaluation protocols: P1 (standard mean Dice including normal cases), P2 (Dice restricted to positive cases), and P3 (the proposed Stratified Evaluation Framework, SEF). Statistical significance is assessed via pairwise Wilcoxon signed-rank tests.

## Results

Phase 1 confirms that any single false-positive pixel on an empty ground-truth yields Dice ≈ 0.0 (smooth = 1e−6), with a maximum theoretical distortion of 0.128 (17.1% relative error) at the BUSI normal-case proportion. Implementation inconsistency is pervasive: the same true-negative scenario returns 1.0 in five implementations but 0.0 in MONAI. Phase 2 simulation shows ranking inversion onset at a 20% normal-case proportion (only 2.9 percentage points above the BUSI actual proportion), with inversion rates reaching 68% at 25% and 100% at 35% normal cases. Phase 3 empirically confirms that 2 of 5 models (U-Net++ and MA-Net) exchange rankings between P1 and P2. SEF further exposes a clinically meaningful trade-off invisible to both protocols: U-Net++ achieves the highest P2 Dice among the competing models (0.781) yet the lowest specificity (0.810), while MA-Net achieves higher specificity (0.857), yielding identical SEF composite scores (0.786) and a distinct clinical safety profile.

## Conclusion

Evaluating segmentation models on datasets containing normal cases using standard mean Dice produces unreliable rankings that fail to reflect true segmentation quality. We propose the Stratified Evaluation Framework (SEF), which decouples segmentation accuracy on positive cases from false-positive control on negative cases into independent, clinically interpretable pathways combined via a prevalence-weighted composite score. SEF is open-source, reproducible (seed = 42; split CSV provided), and adaptable to local clinical prevalence by adjusting the π⁻ parameter. These findings have direct implications for benchmark design and model selection in breast ultrasound AI, particularly for datasets approaching or exceeding 20% normal-case prevalence.

---

**Keywords:** breast ultrasound segmentation, Dice coefficient, evaluation protocol, negative cases, metric distortion, ranking inversion, Stratified Evaluation Framework, SEF, BUSI dataset

**Word count (abstract body):** ~370 words
