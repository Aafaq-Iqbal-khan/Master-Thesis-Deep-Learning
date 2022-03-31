# Master’s Thesis Deep Learning

## Text and Image Based Multi-model Fashion Image Retrieval system
(The repository for the Master’s Thesis based on Deep Learning.)

## Abstract
Interactive image retrieval is an emerging research topic playing a significant role in the success of a wide variety of applications, especially in the fashion domain. However, as fashion product catalogues have grown in size and the number of features each product has increased, it has become more challenging for users to express their needs effectively. In traditional fashion e-stores, users may not be able to specify the details of the outfits more accurately by utilizing a text-based query. Therefore, we focus on a multi-model image retrieval system to integrate the query image with a text query that describes the visual differences between the query image and the search target image. To tackle this task, we investigate a similarity metric between a target image and a candidate image (query image) plus a text query. Both target images and query image are encoded with Efficient-Net and ResNet-50 (only one at a time) into feature vector representation, and encode caption text to a text feature vector using LSTM. Then we compose the query image vector and text feature vector into a single vector which is expected to be biased toward the target image vector with the help of state-of-the-art TIRG. The compositional query-based TIRG achieved a higher average recall with 29.2, than other methods, text-only (21.93), image only Efficient-Net (8.74), and image only Resnet-50 (8.75). The TIRG outperforms text-only, image-only Efficient-Net, and image-only Resnet-50 methods in terms of batch-based classification training loss with values 0.192, 0.42 (65% more), 0.91 (79% more), and 0.52 (63% more) respectively.

## Research Question:
To what extent, integrating both text query and image query help to improve the fashion image retrieval system by investigating a multi-model approach TIRG (Text Image Residual Gating)?

## Objective
The major objectives of the studies are:
1) Compare the performance (recall as major evaluation metric) of the Only-Text based, Only-Image based, and both text and image multi-model based image retrieval system.
2) Based on recall, training loss, and training time performance, evaluate the image retrieval system with different image encoders ResNet-50 and Efficient-Net.

# Fashion IQ Dataset
 ![image](https://user-images.githubusercontent.com/102433874/161109687-40b289f8-a5ea-4a23-bf2d-96aa8fe67f58.png)

# Design Architecture 
![image](https://user-images.githubusercontent.com/102433874/161109709-c71030a6-16d5-48db-a7e9-20ccc3bf2167.png)
 
# Quantitative Results
 
![image](https://user-images.githubusercontent.com/102433874/161109780-42176ec1-e75f-411b-9a40-b75349db060f.png)

![image](https://user-images.githubusercontent.com/102433874/161109812-4d30227d-4554-4d1f-a692-dd8d67d5af12.png)

# Qualitative results
![image](https://user-images.githubusercontent.com/102433874/161109864-26585345-d403-45fe-a483-586806a44fc4.png)
 
# Conclusion and Future Work
Compare the unimodel and multi-model image retrieval system and shows that how results can be improved by using both text and image query
The TIRG outperforms the other methods Text only, Image-only efficient-net, Image-only Resnet-50.
Future work
1)	Integrate our methods with knowledge base concept. 
2)	Extend the applicability of the multi-model image retrieval system to other domains.

 

