![e2jersey](https://github.com/user-attachments/assets/1b343d48-c8d3-4307-9375-998050f9af72)

# E2Jersey: High-Performance ML Jersey Digit Detection System
## Abstract
The E2 Engine is a state-of-the-art jersey number detection system designed to achieve high accuracy while maintaining computational efficiency. Based on the backbone of EfficientNet-B0 and enhanced by knowledge distillation, advanced data augmentation, and learning rate scheduling, the E2 Engine achieves 94.6% accuracy on Epoch 50 for the test dataset and near 90% accuracy on a challenging random internet dataset provided by RoboFlow. The entire E2Jersey Engine is ~16MB in size, which is in part due to our innovations of the E2 system. This article describes the architecture, training methodology, and performance evaluation of the E2 Engine: underlining the major ameliorations compared with past systems. 

The reason why a digit detection engine in DeepField is important is due to the complexity of different nullable characters and possible obstructions in soccer (refs, camera interferers, etc.) as well as the complexity of patterns and colored text/backgrounds in a player's jersey. Therefore, it's important to identify the number correctly. 

## Introduction
The identification of jersey numbers is a crucial task in sport analytics that can enable the analysis of players, performance tracking, and automated event analysis. This is especially useful in sports like soccer, where players move fast and having a dataset is important to assist many aspects of the game. Varying light conditions, occlusions, and diverse jersey designs make this computational task challenging. The E2 Engine addresses challenges by leveraging a backbone in the form of an EfficientNetB0 combined with knowledge distillation and advanced techniques for data augmentation. The system is designed to achieve high accuracy while keeping the model size compact for deployment on edge devices. Similar to professional and expensive VARs (Virtual Assistant Referees), our goal of E2Jersey is to bring many VAR aspects to mobility and full self automation. 

## Architecture
![7ea1f5181b09a42f9673284bdb45a68](https://github.com/user-attachments/assets/74861c5b-cef0-44a1-894c-d688f6d5dcde)
> (Full structure of the neural network) 

DeepField's E2 Engine is based on the EfficientNet-B0 architecture, which is a lightweight and very efficient convolutional neural network (CNN) optimized for image classification tasks. E2 is designed to be performance efficient, adaptable, scalable, and most importantly accurate. **E2's premise is utilizing machine learning to accelerate traditional automated tasks for predictive performance.** 

The main components of the E2 Engine are: 

### 1. EfficientNet-B0 Backbone
EfficientNet-B0 uses a compound scaling method to balance depth, width, and resolution, achieving incredible accuracy with fewer parameters. This is very useful in consideration of mobility and agility of the DeepField system, as fewer parameters result in less file size and less total CPU operations, the performance overhead of checking specific frames for jersey numbers can be reduced to a minimum. The final classification layer is replaced to match the number of jersey number classes in the dataset.

### 2. Knowledge Distillation
A pre-trained ResNet-34 model is used as the teacher model. The student model (EfficientNet-B0) is trained using a combination of:
- Cross-Entropy Loss: For ground truth labels.
- Distillation Loss: For soft targets (predictions from the teacher model). The distillation lossis computed using the Kullback-Leibler (KL) divergence between the student and teacher outputs.

### 3. Advanced Data Augmentation
Heavy random transformations such as horizontal flips, rotation, color jitter, and affine transformations all are applied. These augment the robustness with respect to lighting conditions and changes in general orientations and scales. This way, the model can adapt to determine a jersey number from multiple angles with respect to such transformations. This adds minimal performance overhead as the transformations are minimal subsets and variations of the original trained dataset. Through some number of epochs, the model will gradually adapt to minimize the loss over such dimensions. 

### 4. Learning Rate Scheduling
The ReduceLROnPlateau scheduler adjusts the learning rate during training based on the validation loss. This ensures better convergence and prevents overfitting or vanishing gradients. From this, the training speed is increased without manual supervision, and the file size (and hence the processing required to perform forward propagation) is also reduced due to lowered noise capturing. 

### 5. Early Stopping
Training is halted if the validation loss does not improve for 5 consecutive epochs, preventing gradient problems, noise, and saving computational resources. When Early Stopping is initiated, the E2 system has most likely maximized its use of the data with the given neurons. 

## Training Methodology
The E2 Engine is trained using the following methodology:

### 1. Dataset
The dataset consists of annotated images of players wearing jerseys, split into training, validation, and test sets. A separate random internet dataset is used to evaluate the model's generalization ability. The E2 image classification engine goes through backpropagation to achieve machine learning. 

### 2. Training Process
The model is trained using the AdamW optimizer with a learning rate of 0.0001. Knowledge distillation is applied with a distillation weight of α = 0.5. Data augmentation is applied to the training set to increase diversity. 

### 3. Evaluation
The model is evaluated on both the test dataset and the random internet dataset. Metrics such as accuracy, precision, recall, and F1 score are used to assess performance.

## Results
### 1. Test Set vs. Examination Set Results
The E2 Engine achieves the following results on an Nvidia H800: 
|     | TEST SET | EXAMINATION SET |
| ---- | -------- | -------------- |
| **Accuracy** | 94.6% | 89.0% |
| **Precision** | 93.8% | 88.5% |
| **Recall** | 94.2% | 88.7% | 
| **F1 Score** | 94.0% | 88.6% | 

### 2. Model Size 
The quantized model size is 16.6MB, making it highly efficient and suitable for deployment on edge devices, and especially applicable in DeepField's technologies. 

### 3. Comparison with E1 Engine
The E2 Engine outperforms the E1 Engine in several key areas. The improvements are attributed to:
- The use of EfficientNet-B0 as the backbone
- Knowledge distillation from a ResNet-34 teacher model
- Advanced data augmentation techniques

## Download
[Download our E2Jersey Model (as well as all our past E1 attempts) on HuggingFace. ](https://huggingface.co/DeepFieldML/DeepField_PlayerDigit_Number_Analysis_Engine)
| **Version**         | **Accuracy** | **Size** | **Architecture** | **Changes** |
| -------------       | ------------ | -------  | ---------------  | ----------- |
| DeepField E1 v1     | 87%          | 72.1MB   | E1 CNN           | Applied "E1" series with less overhead and slightly better performance. |
| DeepField E1 v2     | 40%          | N/A (Interrupted) | ViT     | Changed the training technique. |
| DeepField E1 v3     | 68.65%       | 34.3 MB  | Hybrid (CNN + MLP)| Improved hybrid training technique for better perception of data. |
| DeepField E1 v4     | 89.39%       | 94.9 MB  | Revised E1 CNN | Revised the CNN architecture in DeepField E1. |
| E2Jersey v1         | 93.78%       | 16.6 MB  | DeepField E-Series Engine | Pioneered a combination of technologies to achieve optimial performance and operations. |
| E2Jersey v2         | 94.6%        | 16.5 MB  | DeepField E-Series Engine | Revised for larger scale datasets with quantization and better performance. |

![image](https://github.com/user-attachments/assets/73abf287-cea2-4792-a616-ae252b57cec6)

> Note: This repository will only contain the training script. The model itself will always be on HuggingFace. 

## Conclusion
The E2 Engine represents a significant advancement in jersey number detection, achieving 94.6% accuracy on the test dataset and 89% accuracy on a random internet dataset. By leveraging EfficientNet-B0, knowledge distillation, and advanced data augmentation, the E2 Engine outperforms its predecessor while maintaining a compact model size. Future work will explore the integration of Vision Transformers (ViTs) and self-supervised learning to further improve accuracy and generalization.

## Credit
> Created by [Gordon H.](https://www.github.com/ziqian-huang0607) and reviewed by [Will C.](https://www.github.com/willuhd)
> 
1. Robocup Jersey Detection. (2024). Retrieved from https://universe.roboflow.com/robocupjersey-detection/jersey-detection/
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv preprint arXiv:1905.11946.
3. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

## Contact
Please contact us at DeepFieldML@outlook.com if you have any concerns. 
