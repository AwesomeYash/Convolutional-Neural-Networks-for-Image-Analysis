# SVM, MLP and GMM Image Segmentation (MATLAB)

> **Note on the repository name:** despite the name "Convolutional Neural Networks for Image Analysis", this project does **not** contain a convolutional neural network. It compares an RBF-kernel SVM with a small MLP on a synthetic 2-D problem and uses a Gaussian mixture model (GMM) to segment an image. See the *Suggested rename* section.

Graduate machine-learning course project (EECE 5644, Northeastern University, Assignment 4). The full write-up is in [`Report.pdf`](Report.pdf).

## What it does

| Part | Task | Script |
|---|---|---|
| Question 1 | Classify two noisy concentric circles (radius 2 and 4, Gaussian noise σ = 1) with an SVM and an MLP, choosing hyper-parameters by 10-fold cross-validation | `Codes/Ques_1.m` |
| Question 2 | Segment a 321 × 481 RGB image by fitting a Gaussian mixture to pixel colours; the number of components is chosen by 5-fold cross-validated log-likelihood | `Codes/Ques_2.m` |

Supporting file: `Codes/generateConcentricCirclesData.m` (data generator: 1,000 training and 10,000 test samples).

## Methods

- **SVM:** `fitcsvm` with a Gaussian kernel; grid of 12 values of the box constraint C (0.01 to 1000) × 12 kernel scales (0.1 to 31.6) = 144 combinations, each evaluated with 10-fold cross-validation.
- **MLP:** `patternnet` with one hidden layer of radial-basis (`radbas`) units; hidden size chosen from {2, 5, 10, 15, 20, 30, 50, 75, 100} by 10-fold cross-validation.
- **GMM segmentation:** `fitgmdist` on RGB pixel vectors for M = 2 to 6 components (3 replicates per fold, 5 for the final fit), hard assignment of each pixel to its most likely component.

## Results (from the report)

| Experiment | Result |
|---|---|
| SVM | C = 0.0811, kernel scale = 0.8111, CV accuracy 86.00 %, **test accuracy 82.50 %** (8,250 / 10,000) |
| MLP | 5 hidden units, CV accuracy 84.70 %, **test accuracy 83.04 %** (8,304 / 10,000) |
| GMM segmentation | best M = 4 (validation log-likelihood about 2.80 per sample); component shares 18.4 %, 20.7 %, 28.4 %, 32.5 % of 154,401 pixels |

Result figures and command-window screenshots are in `Results/Question 1 Outputs/` and `Results/Question 2 Outputs/`.

## Requirements and how to run

- MATLAB with the **Statistics and Machine Learning Toolbox** (`fitcsvm`, `fitgmdist`) and the **Deep Learning / Neural Network Toolbox** (`patternnet`).
- Run from the `Codes/` folder:

```matlab
Ques_1   % SVM and MLP on concentric circles
Ques_2   % GMM segmentation; reads 299091.jpg
```

`Ques_2.m` reads `299091.jpg` from the current MATLAB path; a copy is in `Results/Question 2 Outputs/`, so add that folder to the path or copy the image next to the script. The other `.jpg` files in the repository root are sample images that the scripts do not use.

## Known limitations

- No random seed is set, so results vary slightly between runs.
- Only single runs are reported (no confidence intervals); the SVM-vs-MLP difference (82.50 % vs 83.04 %) is within what run-to-run variation could explain.
- Report statements not backed by saved outputs: the "theoretical minimum error of about 16 %" for the circles problem and the estimated number of support vectors.
- The report describes hand-written EM with K-means initialisation, but the code uses MATLAB's `fitgmdist`.
- Segmentation uses colour only (no spatial information) and has no ground-truth evaluation.

## Suggested rename

`svm-mlp-gmm-segmentation-matlab` (or similar), so the repository does not suggest CNN experience it does not contain.

## Acknowledgements

Course instructor sample code; the report notes the use of AI assistants for explanations and debugging.

## License

MIT License (see `LICENSE`).
