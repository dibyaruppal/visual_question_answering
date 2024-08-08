#  Visual Question Answering
Constructing a multimodal model with BERT and ViT

## Introduction

This repository provides a novel approach to Visual Question Answering (VQA) by integrating multimodal models using Bidirectional Encoder Representations from Transformers (BERT) and Vision Transformers (ViT). VQA is a complex task that requires a deep understanding of both visual and textual information to provide accurate answers to questions about images. Our proposed model leverages the strengths of BERT in natural language processing and ViT in computer vision to create a unified framework that effectively combines textual and visual data.

## Data

Worked with the VQA Dataset released by Georgia Tech: the Balanced Real Dataset. Due to its large size and computational constraints, we will be using only one-fourth of the total training images. We sampled it by taking random images from the total training images. 
The dataset consists of three components: VQA Annotations, VQA Input Questions, and VQA Input Images. The Annotation Dataset consists of various answers to a particular question about a specific image.


- **Reference :** https://visualqa.org/download.html


## Requirements

- Python == 3.10.13 
- torch == 2.1.2+cpu
- torchvision == 0.16.2+cpu
- numpy == 1.26.4
- pandas == 2.2.2
- transformers == 4.39.3
- Pillow == 9.5.0 
- scikit-learn == 1.2.2
- joblib == 1.4.0 
- Seaborn == 0.12.2
- Matplotlib == 3.7.5

## METHODOLOGY
Designed a Visual Question Answering (VQA) model that integrates both visual and textual features to predict answers to questions about images. The VQAModel model architecture is comprised of the following components:

**Algorithm 1 - VQA Multimodal model:**
1. Vision Transformer: 
    - Utilised a pre-trained Vision Transformer (ViT) model (vit-base-patch16-224-in21k) from Google’s library to extract high-dimensional visual features from input images. The ViT model processes the image and produces a hidden representation. 
2. Text Transformer:
    - For the text component, employed a pre-trained BERT model (bert-base-uncased) to extract semantic features from the input question text. The BERT model encodes the text and provides a contextualised hidden representation.
3. Fully Connected Layers: The outputs from both the vision and text transformers are further processed through fully connected layers:
    - A linear layer (fc1) is defined to process the features from the vision transformer and reduce the dimension of the vision transformer’s hidden state from 768 to an output size of 512.
    - Another linear layer (fc2) is defined to similarly process the hidden state from the text transformer to map to an output size of 512.
    - These processed features are concatenated and these concatenated features(of size 1024) are passed through a final linear layer (fc3) that maps the combined features to the output space corresponding to the number of possible answers
 
![Screenshot 2024-08-09 004014](https://github.com/user-attachments/assets/5893cfc9-6594-4316-9a80-d119277125b2)

**Algorithm 2 - VQA Multimodal model using LoRA:**
1. Vision Transformer:
    - Utilise a pre-trained Vision Transformer (ViT) model (vit-base-patch16-224-in21k) to extract high-dimensional visual features from input images.
     - LoRA (Low-Rank Adaptation) is applied to the Vision Transformer to enable efficient fine-tuning by significantly reducing the number of trainable parameters. Set the LoRA rank to 16.
2. Text Transformer:
    - A pre-trained BERT model (bert-base-uncased) is employed to extract semantic features from input questions.
    - LoRA is similarly applied to the BERT model, with a rank of 16, to facilitate efficient fine-tuning.
3.  Fully Connected Layers: The outputs from both the vision and text transformers are further processed through fully connected layers:
    - A linear layer (fc1) that processes the hidden state output from the Vision Transformer, reducing its dimensionality from 768 to 512
    - Another linear layer (fc2) processes the hidden state output from the BERT model, similarly reducing its dimensionality from 768 to 512.
    - A final linear layer (fc3) that combines the features from both modalities to create features of size 1024 and maps them to the output space, corresponding to the number of possible answers (num answers)

## Result

The visual question-answering (VQA) task involves several key stages, including data preprocessing, model training, and inference, each contributing to the overall performance and accuracy of the model. The metrics experiment involves comparative analysis based on Time taken for Training, Accuracy, F1 score, Precision and Recall.

1. The time taken for training without Lora is 22539.69 seconds while that of LoRA is 13721.88 seconds. This is because LoRA uses fewer trainable parameters, efficiently updates parameters, and preserves pre-trained weights.
![Time_taken_for_training](https://github.com/user-attachments/assets/5f1421ac-cdcb-46c9-9593-8f7629dfc3bb)\

2. Visualisation of the training curve for both models
![acc_without_lora](https://github.com/user-attachments/assets/e819b171-a92a-4fe5-8d75-ec0bbce91def)  ![acc_with_lora](https://github.com/user-attachments/assets/f094b1b4-df7c-4bb1-9404-d7d3032e9c31)

3. Performance for Different Metrics for both the Models
![performance_metrics](https://github.com/user-attachments/assets/5cd2cd3e-5992-402b-a906-7bf1648aba4d)

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Make sure to include tests if you are adding new functionality.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or suggestions, please open an issue in the repository or contact the project maintainer at [Dibyarup Pal](mailto:dibyarup.pal@iiitb.ac.in).

---

Thank you for using the Visual Question Answering solution. I hope you find it useful and easy to use!
