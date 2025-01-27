# Image Enhancement Script

This Python script enhances images through a multi-stage pipeline, improving their visual quality. The pipeline includes:

*   Brightness and saturation adjustment
*   White balancing
*   Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
*   Denoising

## Example Results

The table below shows examples of the enhancement pipeline's effects on an image, demonstrating the impact of different parameter combinations.

<table style="width:100%; text-align: center;">
  <tr>
    <th style="text-align: center;">Original</th>
    <th style="text-align: center;">Brightness: 20, Saturation: 1.5, White Balance: Percentile</th>
  </tr>
  <tr>
    <td><img src="assets/alhambra1_b0_s1_none_noclahe.jpg" alt="Original" width="400"></td>
    <td><img src="assets/alhambra1_b20_s1.5_percentile_noclahe.jpg" alt="B: 20, S: 1.5, Percentile" width="400"></td>
  </tr>
    <tr>
    <th style="text-align: center;">Brightness: 20, Saturation: 1.5, White Balance: Percentile, with CLAHE</th>
    <th style="text-align: center;">Brightness: 20, Saturation: 1.5, White Balance: Gray World, with CLAHE</th>
  </tr>
  <tr>
    <td><img src="assets/alhambra1_b20_s1.5_percentile_clahe.jpg" alt="B: 20, S: 1.5, Percentile, w/ CLAHE" width="400"></td>
    <td><img src="assets/alhambra1_b20_s1.5_grayworld_clahe.jpg" alt="B: 20, S: 1.5, Gray World, w/ CLAHE" width="400"></td>
  </tr>
    <tr>
      <th colspan="2" style="text-align: center;">Final Enhanced Image</th>
    </tr>
        <tr>
      <td colspan="2" style="text-align: center;"><img src="assets/alhambra_enhanced.jpg" alt="Final Enhanced Image" width="400"></td>
    </tr>
</table>

The final enhanced image above was generated by further processing the "Gray World with CLAHE" image with the following parameters:

*   Brightness factor: 10
*   CLAHE kernel size: 20
*   Denoising applied
