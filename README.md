# Breast Cancer Classification Model
This is a simple deep learning classification model for the classification of histopathological images of breast cancer.
## Data
The dataset is an open breast cancer classification dataset 
(https://www.kaggle.com/paultimothymooney/breast-histopathology-images/data).
162 whole slide images where divided into 277,524 image patches of size 50 x 50 (198,738 IDC negative and 78,786 IDC positive).
The dataset was split into training (70%), validation (10%) and test (20%) set. The script for splitting can be found in
./src/sample_subsets.py.  

## Model experiments
Two models where tested for this task.
### Mobilenetv2 Backbone
The backbone of this model is mobilenetv2 with a custom classification head. The network was pretrained on imagenet and
is taken from keras applications.
The model was so far just trained for 60 epochs with 1000 batches per epoch because of heavy hardware limitations 
(trained on a laptop CPU for ~18h). Here you see the training and validation accuracy:
![training](demo_images/train_mobilenetv2.png)
The validation accuracy is unstable because it evaluates on 100 random batches of the validation set (else it would take too long).
The network has not converged yet but was trained to an accuracy of ~90%.

Here are some example images with ground truth and network predictions (0=no cancer, 1=cancer):
![](demo_images/mobilenet_output/1.jpg)
![](demo_images/mobilenet_output/2.jpg)
![](demo_images/mobilenet_output/3.jpg)
![](demo_images/mobilenet_output/4.jpg)
![](demo_images/mobilenet_output/5.jpg)
![](demo_images/mobilenet_output/6.jpg)
![](demo_images/mobilenet_output/7.jpg)
![](demo_images/mobilenet_output/8.jpg)
![](demo_images/mobilenet_output/9.jpg)

### Simple CNN
For comparison a very simple and lightweight CNN consisting of several separable convolution blocks was trained.
The training progress of 70 epochs is shown here:
![](demo_images/train_simple_cnn.png)
As seen in the figure, also this network was not trained until full convergence even though it trains
much faster than the above model.
Some sample predictions are shown here:
![](demo_images/simple_cnn_output/10.png)
![](demo_images/simple_cnn_output/11.png)
![](demo_images/simple_cnn_output/12.png)
![](demo_images/simple_cnn_output/13.png)
![](demo_images/simple_cnn_output/14.png)

## Usage
To make this code run on your machine you need to:
* download the dataset from https://www.kaggle.com/paultimothymooney/breast-histopathology-images/data
* set up a virtual environment and activate it
* pip install the requirements: `pip install -r requirements.txt`
* Run the network: 
    * To see the usage options `python src/main.py -h`
    * please make sure you define the correct dataset path