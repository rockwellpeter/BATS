**Codes for CVPR2025**<br>Our core codes are available in Desktop/codes.
<br>
[Overall review (PDF)](https://github.com/user-attachments/files/17758956/overall.review.pdf)

**Dependencies：**
<br>cuda: 11.7
<br>cudnn: 9.1.0
<br>python 3.9
<br>pytorch >= 1.13.0

**Results：**
| Method          | Dataset   | Super Resolution |       |        |       | Dataset       | Face Inpainting |       |       |       |
| --------------- | --------- | ---------------- | ----- | ------ | ----- | ------------- | ---------------- | ----- | ----- | ----- |
|                 |           | PSNR↑           | SSIM↑ | LPIPS↓ | FID↓  |               | PSNR↑           | SSIM↑ | LPIPS↓| FID↓  |
| CNN-Baseline    | DIV2K     | 26.64           | 0.6729| 0.3890 | 133.95| CelebA-HQ     | 29.22           | 0.9218| 0.0650| 38.35 |
| DDPM            |           | 25.40           | 0.7326| 0.1859 | 19.64 |               | 26.26           | 0.7555| 0.1166| 24.23 |
| DDPM+♦          |           | 26.11           | 0.7439| 0.2044 | 19.01 |               | 26.39           | 0.8977| 0.0695| 52.02 |
| DDPM+①         |           | **28.02**       | **0.7834** | 0.2413 | 19.29 |           | 28.73           | **0.9152**| **0.0633** | 47.14 |
| DDPM+②         |           | 26.55           | 0.7692 | 0.2056 | **18.80** |         | 26.67           | 0.8859| 0.0707 | 55.98 |
| **Ours**        |           | **28.40**       | **0.8111** | 0.2466 | **18.01** |      | **30.76**       | **0.9383**| **0.0496** | **34.39** |

| Method          | Dataset   | Raindrop Removal |       |        |       | Dataset       | Image Denoising |       |       |       |
| --------------- | --------- | ---------------- | ----- | ------ | ----- | ------------- | ---------------- | ----- | ----- | ----- |
|                 |           | PSNR↑           | SSIM↑ | LPIPS↓ | FID↓  |               | PSNR↑           | SSIM↑ | LPIPS↓| FID↓  |
| CNN-Baseline    | RainDrop  | 27.92           | 0.8456| 0.1165 | 71.31 | CBSD68        | 27.63           | 0.7556| 0.2715| 110.79 |
| DDPM            |           | 28.47           | 0.8823| 0.0942 | 68.20 |               | 26.03           | 0.7344| **0.2229** | 98.63 |
| DDPM+♦          |           | 28.10           | **0.8971** | **0.0992** | 63.46 |        | 25.50           | 0.7238| 0.2565 | 132.12 |
| DDPM+①         |           | **28.66**       | 0.8965| 0.0978 | **59.63** |           | **26.59**       | 0.7422| 0.2617 | **97.78** |
| DDPM+②         |           | 28.26           | 0.8837| 0.0934 | 55.32 |               | 26.82           | 0.7527| 0.2664 | 103.16 |
| **Ours**        |           | **29.13**       | **0.9006** | **0.0733** | **49.56** |    | **28.16**       | **0.8053**| 0.2123 | **95.32** |

Table 1. Performance comparisons across different methods on multiple image restoration tasks. ♦ represents the training method proposed in DREAM, and ① represents our distribution extend training method proposed, while ② represents our distribution alignment training method proposed.
