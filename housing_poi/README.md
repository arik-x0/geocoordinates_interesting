
File	Role
model.py	HousingEdgeCNN — multi-scale edge detection CNN
utils.py	NDBI computation, label generation, scoring, visualisation
dataset.py	EuroSAT loader with auto-generated structure labels
train.py	Training loop with DiceBCELoss
predict.py	Inference, density ranking, visualisation
requirements.txt	Dependencies
Model — HED-inspired multi-scale edge CNN


Stage 1: ConvBNReLU(3→32)   × 2, d=1  → 64×64  fine local edges
Stage 2: Pool + Conv(32→64)  × 2, d=1  → 32×32  building-scale features
Stage 3: Pool + Conv(64→128) × 2, d=1  → 16×16  block-scale patterns
Stage 4: Conv(128→256)       × 2, d=2  → 16×16  wide context (dilation, no pool)
Side output at each stage → upsample to 64×64
Fusion: Cat(4) → Conv(4→1) → Sigmoid
The dilated stage 4 expands the receptive field to see building-block-level context without losing spatial resolution, which is important for detecting rectangular building outlines.

Label generation (no manual annotations needed)

Ground truth is auto-derived per image from the 13 Sentinel-2 bands:

NDBI (Normalized Difference Built-up Index) = (SWIR − NIR) / (SWIR + NIR) — positive values indicate rooftops, pavements, roads
Sobel gradient on grayscale RGB — reinforces sharp man-made edges
Combined score thresholded at 0.5 → binary mask
Morphological closing (3×3 kernel) fills building interiors
Scoring and density filter

housing_score = fraction of pixels predicted as structure (threshold 0.5)
Low-density residential: 5% ≤ score ≤ 20% → the target POI zone
predict.py partitions all test images into three groups and ranks the low-density bucket by proximity to the midpoint (12.5%)
Usage:


```cd housing_poi
python train.py --data-dir data --checkpoint-dir checkpoints --epochs 25
python predict.py --checkpoint checkpoints/best_model.pth --output-dir output --top-n 20```