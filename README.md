# DenseNet-COCO
This project contains the training and evaluation of different Densenet models on the MS-COCO dataset


  DenseNet (Densely Connected Convolutional Networks) represents a significant advance in the architecture of convolutional neural networks, providing better gradient propagation and feature reuse. This is achieved through dense connectivity between layers, where each layer receives as input all the outputs of previous layers.\
    MS-COCO (Microsoft Common Objects in Context) is an extensive and diverse dataset widely used for training and evaluating object recognition models. It includes over 118,000 images in the training set, 5,000 in the validation set and 41,000 in the test set, with 80 varied object categories. The objectives of this work are to investigate the classification performance of the DenseNet architecture on the MS-COCO dataset, with implementation in PyTorch.\
    The models were trained using the MS-COCO subset "train 2017" and tested on the validation subset, "val2017", observing the results in terms of accuracy both on individual classes and on the whole set. Over 20 training epochs, the performance evolution will be analyzed in order to better understand the capabilities and limitations of these architectures in the context of the data provided by MS-COCO.\
    I observed the behavior of DenseNet models 121 and 169, trained over up to 20 epochs, then compared the general accuracy, as well as the accuracy on specific classes and showed the confusion matrices for the models trained for the chosen maximum of epochs. In addition, I also compared the evolution of the loss over the training epochs, observing generally better results for the DenseNet169 model, but behaving quite similarly.

  Results:

  The graphs for both models showing the accuracy depending on the number of epochs:

![Accuracy121](https://github.com/user-attachments/assets/0ac4f1b0-9146-499b-b482-84621031ea14)
![Accuracy169](https://github.com/user-attachments/assets/091e9aae-b894-40da-aba2-a80d22fbf350)

  The graph of the losses of the 2 models during the 20 epochs:
![LossComparison](https://github.com/user-attachments/assets/e9202017-6de0-4218-97f3-6acb25bea24d)

  The confusion matrices for both models trained for 20 epochs:
  ![denseNet121-confMat](https://github.com/user-attachments/assets/499b8ab1-797b-4f3e-9430-2ee71e051f05)
![denseNet169-confMat](https://github.com/user-attachments/assets/fa48eda0-b53f-487d-9c9b-be428c2c70e1)

  In conclusion, DenseNet-121 and DenseNet-169 are neural network architectures used for object recognition, performing well on the COCO dataset. DenseNet-121, with 121 layers, is more computationally efficient and easier to train, but exhibits significant variability in accuracy across individual classes, indicating the need for further tuning.\
    The multi-layered DenseNet-169 offers higher representational capacity and better performance on complex classes, but requires larger resources and longer training time. Its accuracy is more stable than DenseNet-121, although fluctuations persist.\
    The choice between the two architectures depends on resource constraints and application requirements: DenseNet-121 is better suited for efficiency, while DenseNet-169 offers a marginal advantage in recognizing complex features.
