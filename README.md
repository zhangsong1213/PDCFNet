# PDCFNet: Enhancing Underwater Images through Pixel Differential Convolution and Cross-Level Feature Fusion
The majority of deep learning approaches utilize traditional vanilla convolution for modeling enhanced images. Vanilla convolution excels in capturing local features and learning the spatial hierarchical structure of images. However, it tends to smooth the input image, which can somewhat limit feature expression and modeling. A characteristic trait of underwater degraded images is blurriness, and the enhancement goal is to make textures and details (high-frequency features) more visible in the image. Hence, leveraging high-frequency features could improve enhancement performance. To this end, we introduce Pixel-Difference Convolution (PDC) to focus on areas with significant gradient changes in the image, facilitating better modeling of enhanced images. We propose an underwater image enhancement network based on PDC and cross-level feature fusion, named PDCFNet. Specifically, we designed a detail enhancement module using parallel PDC to capture high-frequency features, achieving superior enhancement of details and textures. The designed cross-level feature fusion module concatenates, multiplies, and performs other operations on features from different layers, ensuring extensive interaction and enhancement among various features. Our proposed PDCFNet has achieved the best performance to date on the UIEB dataset, with a PSNR of 27.37 and an SSIM of 92.02.


## Train the Model
python main.py

## Test the Model
python eval.py

## Environment
environment.txt
