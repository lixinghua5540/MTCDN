# MTCDN
Paper Ttile: Concatenated Deep Learning Framework for Multi-task Change Detection of Optical and SAR Images<br>

Reference: Z. Du, X. Li, J. Miao, Y. Huang, H. Shen and L. Zhang, “Concatenated Deep Learning Framework for Multi-task Change Detection of Optical and SAR Images,” IEEE Journal of Selected Topics in Applied Earth Observation and Remote Sensing, vol. , pp. , , 2023.

##***Introduction***<br>

Optical and synthetic aperture radar (SAR) images provide complementary information to each other. However, the heterogeneity of same-ground objects brings a large difficulty to change detection (CD). Correspondingly, transformation-based methods are developed with two independent tasks of image translation and CD. Most methods only utilize deep learning for image translation, and the simple cluster and threshold segmentation leads to poor CD results. Recently, DTCDN was proposed to apply deep learning for image translation and CD to improve the results. However, DTCDN requires the sequential training of the two independent subnetwork structures with a high computational cost. Towards this end, a concatenated deep learning framework, multi-task change detection network (MTCDN), of optical and SAR images is proposed by integrating change detection network into a complete generative adversarial network (GAN). This framework contains two generators and discriminators for optical and SAR image domains. Multi-task refers to the combination of image identification by discriminators and CD based on an improved UNet++. The generators are responsible for image translation to unify the two images into the same feature domain. In the training and prediction stages, an end-to-end framework is realized to reduce cost. The experimental results on four optical and SAR datasets prove the effectiveness and robustness of the proposed framework over eight baseline.<br>

![Overall](https://github.com/lixinghua5540/MTCDN/assets/75232301/302aa7b1-e70b-46f5-8036-e4e574eaa8e9)<br>

##***Usage***<br>
The implementation code of image translation and change detection are integrated as a whole: 
after preprocess of the image, you can directly run the python file named train_MTCDN.py, the checkpoints will be saved in the fold named checkpoints. 
As MTCDN are will trained, you can run test_MTCDN.py to get the prediction of change detection results and image translation results.
The Training data and testing data will be available soon.
