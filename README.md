# Image Enhancement Pipeline for UXO Analysis  

A Python script designed to enhance images of unexploded ordnance (UXO) through a configurable multi-stage pipeline. Optimizes visual quality for downstream processing tasks like object detection or classification.

---

## Features  
- **White Balance Correction** (`grayworld`, `percentile`, or `lab`)  
- **Brightness/Saturation Adjustment**  
- **Local Contrast Enhancement (LCE)**  
- **Denoising** (Bilateral Filtering)  
- **Tonemapping** (Spatial or LECARM)  
- **CLAHE** (Contrast-Limited Adaptive Histogram Equalization)  
- **Parameter-Driven Customization**  

---

## Example Results  

### Pipeline Progression  
| Stage | Result |  
|-------|--------|  
| **Original Image** | ![Original](images/uxo1.jpg) |  
| **White Balance (Grayworld)** | ![Grayworld](assets/uxo1-wb_grayworld.jpg) |  
| **+ Denoising & LCE** | ![Denoise + LCE](assets/uxo1-wb_grayworld_denoise-d5-sc50.0-ss50.0_lce-degree3-smoothing0.2.jpg) |  
| **+ LECARM Tonemapping** | ![Full Pipeline](assets/uxo1_b10.0_s1.25-wb_grayworld_denoise-d5-sc50.0-ss50.0_lce-degree3-smoothing0.2_lecarm-camerasigmoid-down1.0-scale0.8.jpg) |  

**Configuration Used for Final Enhanced Image**:

```yaml  
brightness_factor: 10.0
saturation_factor: 1.25

white_balance:
  enabled: True
  algorithm: grayworld
  percentile: 99.0

denoising:
  enabled: True
  diameter: 5
  sigma_color: 50.0
  sigma_space: 50.0

local_contrast_enhancement:
  enabled: True
  degree: 3
  smoothing: 0.2

lecarm_tonemapping:
  enabled: True
  camera_model: sigmoid
  downsampling: 1.0
  scaling: 0.8
```

## Comparison of differing methods

<table style="width:100%; text-align: center;">
  <tr>
    <th style="text-align: center;">Clahe</th>
    <th style="text-align: center;">Local Contrast Enhancement</th>
  </tr>
  <tr>
    <td><img src="assets/uxo1_b10.0_s1.25-wb_grayworld_clahe-kernel3-clip1.0_denoise-d5-sc50.0-ss50.0.jpg" alt="Clahe" width="400"></td>
    <td><img src="assets/uxo1-wb_grayworld_denoise-d5-sc50.0-ss50.0_lce-degree3-smoothing0.2.jpg" alt="Local Contrast Enhancement" width="400"></td>
  </tr>
  <tr>
    <th style="text-align: center;">LECARM Tonemapping</th>
    <th style="text-align: center;">Spatial Tonemapping</th>
  </tr>
  <tr>
    <td><img src="assets/uxo1_b10.0_s1.25-wb_grayworld_denoise-d5-sc50.0-ss50.0_lce-degree3-smoothing0.2_lecarm-camerasigmoid-down1.0-scale0.8.jpg" alt="LECARM Tonemapping" width="400"></td>
    <td><img src="assets/uxo1_b10.0_s1.25-wb_grayworld_denoise-d5-sc50.0-ss50.0_lce-degree3-smoothing0.2_stm-smoothing0.2-mid_tone0.5-tonal_width0.5-areas_dark0.5-areas_bright0.5.jpg" alt="Spatial Tonemapping" width="400"></td>
  </tr>
</table>

Local (spatial) tonemapping offers computational efficiency and superior performance in high dynamic range (HDR) scenes containing extreme luminance variations (e.g., simultaneous bright highlights and dark shadows). However, its edge-preserving smoothing operations inherently:  
1. **Reduce micro-texture detail** through regional averaging  
2. **Amplify existing noise** in low-contrast areas due to local contrast boosting  

Global methods like LECARM better preserve original texture structures through whole-image dynamic range compression, but sacrifice localized contrast adaptation - making them less effective for scenes requiring simultaneous highlight/shadow recovery.  

**Practical Trade-off**:  
- *Choose spatial tonemapping* for HDR recovery in resource-constrained environments  
- *Prefer global methods* when texture fidelity is critical and lighting is relatively uniform  

## Pipeline Order Rationale  

Enhancements are applied in this sequence to minimize artifacts and dependencies:

1. **Denoising**
2. **White Balance**  
3. **Brightness Adjustment**  
4. **Local Contrast Enhancement**  
5. **CLAHE**
6. **Spatial Tonemapping**
7. **LECARM Tonemapping**  
8. **Saturation Adjustment**  

*Why this order?* Early denoising prevents noise amplification. Color corrections (white balance) precede luminance adjustments, while saturation is finalized last to compensate for desaturation from prior steps.
