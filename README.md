# FORECASS: Forgetting-Resilient Continuous Source-Free UDA for Semantic Segmentation

## Abstract
Semantic segmentation is fundamental for autonomous driving, requiring dense scene understanding across diverse and continu-
ously changing environments. Real-world deployment must overcome two major challenges: (1) dynamic environmental shifts due
to weather, lighting, and geography, and (2) the inability to retain labeled source data for continual adaptation. To address these
issues, we propose a novel source-free Adaptive and Forgetting-Resilient Continual Unsupervised Domain Adaptation for Semantic
Segmentation (FORECASS). We introduce a teacher–student based framework, using EMA (Exponential Moving Average) updat-
ing technique, to produce stable pseudo-labels during continual adaptation. Central to our method is a refiner-based error estimation
model that predicts pixel-wise pseudo-label reliability during adaptation. By leveraging the error map, the model selectively fo-
cuses learning on uncertain and challenging regions, playing a critical role in mitigating catastrophic forgetting. Complementarily,
a structure-aware consistency mechanism enforces semantic coherence across views, further enhancing stability during sequential
adaptation. Lightweight knowledge distillation is also incorporated to smooth alignment between the pseudo-label generator and
the adaptation model. We validate our framework on the challenging continual adaptation sequence GTA → Cityscapes → IDD →
Mapillary, achieving state-of-the-art results over both source-dependent and source-free baselines.
