# PDCFNet: Enhancing Underwater Images through Pixel Differential Convolution and Cross-Level Feature Fusion
Majority of deep learning methods utilize vanilla convolution for enhancing underwater images. While vanilla convolution excels in capturing local features and learning the spatial hierarchical structure of images, it tends to smooth input images, which can somewhat limit feature expression and modeling. A prominent characteristic of underwater degraded images is blur, and the goal of enhancement is to make the textures and details (high-frequency features) in the images more visible. Therefore, we believe that leveraging high-frequency features can improve enhancement performance. To address this, we introduce Pixel Difference Convolution (PDC), which focuses on gradient information with significant changes in the image, thereby improving the modeling of enhanced images. We propose an underwater image enhancement network, PDCFNet, based on PDC and cross-level feature fusion. Specifically, we design a detail enhancement module based on PDC that employs parallel PDCs to capture high-frequency features, leading to better detail and texture enhancement. The designed cross-level feature fusion module performs operations such as concatenation and multiplication on features from different levels, ensuring sufficient interaction and enhancement between diverse features. Our proposed PDCFNet achieves a PSNR of 27.37 and an SSIM of 92.02 on the UIEB dataset, attaining the best performance to date.

# Our proposed PCDFNet
![DCFNet](https://github.com/user-attachments/assets/6110e497-4471-48b0-a2a4-858a71892711)

# Video Display
【PCDFNet】 https://www.bilibili.com/video/BV1wHxLeKEhU/?share_source=copy_web&vd_source=7aec8b91ccee858c79e9d772a3d42e3b

## Compared to other methods
![Compared to other methods](https://github.com/user-attachments/assets/0e93b54c-bf47-4569-b7bb-d13fcd0b5a26)

## Train the Model
python main.py

## Test the Model
python eval.py

## Environment
environment.txt
