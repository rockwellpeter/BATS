**Codes for CVPR2025**<br>Our core codes are available in Desktop/codes.
<br>
[Overall review (PDF)](https://github.com/user-attachments/files/17758956/overall.review.pdf)

**Dependencies：**
<br>cuda: 11.7
<br>cudnn: 9.1.0
<br>python 3.9
<br>pytorch >= 1.13.0


\begin{table*}[t]
\small
\centering
\begin{tabular}{ccccccccccc}
\toprule
\multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Dataset}} & \multicolumn{4}{c}{\textbf{Super Resolution}} & \multirow{2}{*}{\textbf{Dataset}} & \multicolumn{4}{c}{\textbf{Face Inpainting}} \\ 
\cmidrule(lr){3-6} \cmidrule(lr){8-11} 
& & \textbf{PSNR↑} & \textbf{SSIM↑} & \textbf{LPIPS↓} & \textbf{FID↓} & & \textbf{PSNR↑} & \textbf{SSIM↑} & \textbf{LPIPS↓} & \textbf{FID↓} \\ 
\midrule
\rowcolor{gray!20}CNN-Baseline & \multirow{5}{*}{DIV2K} & 26.64 & 0.6729 & 0.3890 & 133.95 & \multirow{5}{*}{CelebA-HQ} & {\color[HTML]{3E69BB} \textbf{29.22}} & {\color[HTML]{3E69BB} \textbf{0.9218}} & 0.0650 & 38.35 \\
DDPM & & 25.40 & 0.7326 & {\color[HTML]{FF0000} \textbf{0.1859}} & 19.64 & & 26.26 & 0.7555 & 0.1166 & {\color[HTML]{FF0000} \textbf{24.23}} \\
DDPM+\ding{169} & & 26.11 & 0.7439 & {\color[HTML]{3E69BB} \textbf{0.2044}} & 19.01 & & 26.39 & 0.8977 & 0.0695 & 52.02 \\
DDPM+\ding{172} & & {\color[HTML]{3E69BB} \textbf{28.02}} & {\color[HTML]{3E69BB} \textbf{0.7834}} & 0.2413 & 19.29 & & 28.73 & 0.9152 & {\color[HTML]{3E69BB} \textbf{0.0633}} & 47.14 \\
DDPM+\ding{173} & & 26.55 & 0.7692 & 0.2056 & \color[HTML]{3E69BB}\textbf{18.80} & & 26.67 & 0.8859 & 0.0707 & 55.98 \\
\rowcolor{gray!20} Ours & & {\color[HTML]{FF0000} \textbf{28.40}} & {\color[HTML]{FF0000} \textbf{0.8111}} & 0.2466 & {\color[HTML]{FF0000} \textbf{18.01}} & & {\color[HTML]{FF0000} \textbf{30.76}} & {\color[HTML]{FF0000} \textbf{0.9383}} & {\color[HTML]{FF0000} \textbf{0.0496}} & {\color[HTML]{3E69BB} \textbf{34.39}} \\ 
\midrule \midrule
\multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Dataset}} & \multicolumn{4}{c}{\textbf{Raindrop Removal}} & \multirow{2}{*}{\textbf{Dataset}} & \multicolumn{4}{c}{\textbf{Image Denoising}} \\ 
\cmidrule(lr){3-6} \cmidrule(lr){8-11} 
& & \textbf{PSNR↑} & \textbf{SSIM↑}& \textbf{LPIPS↓} & \textbf{FID↓} & & \textbf{PSNR↑} & \textbf{SSIM↑} & \textbf{LPIPS↓} & \textbf{FID↓} \\ 
\midrule
\rowcolor{gray!20}CNN-Baseline & \multirow{5}{*}{RainDrop} & 27.92 & 0.8456 & 0.1165 & 71.31 & \multirow{5}{*}{CBSD68} & 27.63 & 0.7556 & 0.2715 & 110.79 \\
DDPM & & 28.47 & 0.8823 & {\color[HTML]{3E69BB} \textbf{0.0942}} & 68.20 & & 26.03 & 0.7344 & {\color[HTML]{3E69BB} \textbf{0.2229}} & 98.63 \\
DDPM+\ding{169} & & 28.10 & {\color[HTML]{3E69BB} \textbf{0.8971}} & 0.0992 & 63.46 & & 25.50 & 0.7238 & 0.2565 & 132.12 \\
DDPM+\ding{172} & & {\color[HTML]{3E69BB} \textbf{28.66}} & 0.8965 & 0.0978 & {\color[HTML]{3E69BB} \textbf{59.63}} & & {\color[HTML]{3E69BB} \textbf{26.59}} & {\color[HTML]{3E69BB} \textbf{0.7422}} & 0.2617 & {\color[HTML]{3E69BB} \textbf{97.78}} \\
DDPM+\ding{173} & & 28.26 & 0.8837 & 0.0934 & 55.32 & & 26.82 & 0.7527 & 0.2664 & 103.16 \\
\rowcolor{gray!20} Ours & & {\color[HTML]{FF0000} \textbf{29.13}} & {\color[HTML]{FF0000} \textbf{0.9006}} & {\color[HTML]{FF0000} \textbf{0.0733}} & {\color[HTML]{FF0000} \textbf{49.56}} & & {\color[HTML]{FF0000} \textbf{28.16}} & {\color[HTML]{FF0000} \textbf{0.8053}} & {\color[HTML]{FF0000} \textbf{0.2123}} & {\color[HTML]{FF0000} \textbf{95.32}} \\ 
\bottomrule
\end{tabular}
\caption{Performance comparisons across different methods on multiple image restoration tasks. \color[HTML]{FF0000}{\textbf{Bold red text }}\color{black} and \color[HTML]{3E69BB}{\textbf{blue bold text}} \color{black} indicate the best and second-best performance, respectively.}
\label{table:combined_results}
\end{table*}

