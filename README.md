# NAFNet: Image Deblurring on the GoPro Dataset

## 1. Project Overview
This project implements and evaluates NAFNet (Nonlinear Activation Free Network) for image deblurring using the GoPro dataset.
The objective is to restore sharp images from motion-blurred inputs using a deep learning model.
The implementation uses the official NAFNet repository and BasicSR framework, running on Google Colab with an NVIDIA Tesla T4 GPU.

## 2. Technologies Used
- Python
- PyTorch
- BasicSR
- OpenCV
- NumPy
- Google Colab
- GoPro Dataset

## 3. Model Architecture
Model: NAFNet  
Architecture: NAFNetLocal  
NAFNet is an image restoration network designed to recover high-quality images from degraded inputs.  
BasicSR provides the official model implementation along with utilities for dataset loading, training, checkpoint management, and evaluation.  
Official repository: https://github.com/megvii-research/NAFNet

## 4. Dataset
We use the GoPro dataset, which contains paired blurry and sharp images.  
Each scene contains:
- blur – Motion-blurred input images.
- sharp – Corresponding sharp ground-truth images.
- blur_gamma – Not used in this experiment.

### Dataset statistics

| Split | Image pairs |
| :--- | :--- |
| Training | 1,605 |
| Testing | 379 |

### Training preprocessing
The training images are organized into separate input and target directories.  
The official GoPro preprocessing script is executed to extract 256 × 256 patches.  
This produces:
- 12,840 cropped training entries.
- Blurry and sharp crop directories.
- LMDB databases for efficient data loading.

### Test preprocessing
The test images are organized into input and target directories and converted into LMDB databases.  
Unlike training images, test images are not cropped.

## 5. Training Configuration
The original NAFNet GoPro configuration is modified to accommodate the Tesla T4 GPU.

| Parameter | Value |
| :--- | :--- |
| GPU | NVIDIA Tesla T4 |
| Architecture | NAFNetLocal |
| Width | 32 |
| Encoder blocks | [1, 1, 1, 28] |
| Middle blocks | 1 |
| Decoder blocks | [1, 1, 1, 1] |
| Batch size | 1 |
| Crop size | 256 × 256 |
| Iterations | 2,000 |
| Optimizer | AdamW |
| Loss function | PSNRLoss |
| Checkpoint frequency | 500 iterations |

The architecture settings, loss function, and optimizer are retained from the original configuration. The batch size and training schedule are adjusted for the available GPU resources.  
Automatic validation is disabled because the single-GPU validation hook encountered a distributed-process error. Evaluation is performed separately after training.

## 6. Model Training
Training is performed using the BasicSR training script.  
The model learns to restore sharp images by comparing its predictions against the corresponding ground-truth images.  
The training process includes:
1. Loading the training dataset.
2. Initializing the NAFNet architecture.
3. Initializing the PSNR loss and AdamW optimizer.
4. Performing forward propagation.
5. Calculating the loss.
6. Performing backpropagation.
7. Updating model weights.
8. Saving checkpoints periodically.

A total of 2,000 optimizer updates are performed.

## 7. Model Checkpoints
The trained model checkpoints are saved in:  
`experiments/NAFNet-GoPro-width32-T4/models/`

Generated checkpoints:
- net_g_500.pth
- net_g_1000.pth
- net_g_1500.pth
- net_g_2000.pth
- net_g_latest.pth

The final checkpoint, net_g_2000.pth, is used for inference and evaluation.

## 8. Inference
The trained checkpoint is loaded into the same NAFNet architecture used during training.  
The inference process involves:
1. Loading the trained checkpoint.
2. Switching the model to evaluation mode.
3. Reading a blurry test image.
4. Converting the image into a normalized PyTorch tensor.
5. Passing the tensor through NAFNet.
6. Generating the restored image.

Inference is performed using `torch.no_grad()` to avoid unnecessary gradient calculations.

## 9. Evaluation Metrics
The model is evaluated on all 379 test images using two metrics.

### PSNR (Peak Signal-to-Noise Ratio)
Measures pixel-level reconstruction quality. Higher values indicate better reconstruction.

### SSIM (Structural Similarity Index)
Measures structural similarity between the restored image and ground truth. Higher values indicate better structural similarity.

## 10. Experimental Results
Results obtained after 2,000 training iterations:

| Metric | Blurry Input | NAFNet Output |
| :--- | :--- | :--- |
| Average PSNR | 26.4146 dB | 26.6398 dB |
| Average SSIM | 0.8285 | 0.8304 |

Additional observations:
- PSNR improved for 366 out of 379 test images.
- PSNR gain: +0.2252 dB.
- SSIM gain: +0.0019.

The experiment demonstrates an improvement in image restoration quality after training.  
The improvement is modest because this experiment uses a shortened 2,000-iteration training schedule. These results should not be interpreted as the full training performance of NAFNet.

## 11. Visual Comparison
The notebook includes a visual comparison of three images:
1. Blurry Input – Original motion-blurred image.
2. NAFNet Restored – Image generated by the trained model.
3. Sharp Ground Truth – Original sharp reference image.

This comparison helps evaluate the restoration quality visually.

## 12. Output Directories

| Output | Location |
| :--- | :--- |
| Training configuration | `options/train/GoPro/NAFNet-width32-T4.yml` |
| Model checkpoints | `experiments/NAFNet-GoPro-width32-T4/models/` |
| Restored images | `results/NAFNet-GoPro-T4/` |

## 13. How to Run
1. Open the notebook in Google Colab.
2. Select a GPU runtime.
3. Upload the GoPro dataset to Google Drive.
4. Ensure the dataset follows the directory structure described above.
5. Execute the notebook cells sequentially.
6. Authorize Google Drive access when prompted.
7. Run the training process.
8. Execute the inference and evaluation cells.

## Reference
NAFNet Official GitHub Repository: https://github.com/megvii-research/NAFNet
