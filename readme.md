# FaceExpressionRecognition_PaddlePaddle

Introduction: Face Expression Recognition Implemented by Baidu's PaddlePaddle

1. Datasets: CK+ database: You can go to the [official website](http://www.consortium.ri.cmu.edu/ckagree/) to download Emotion_labels.zip and extend-cohn-kanade-images.zip.
 You also can go to the [Baidu network disk](https://download.csdn.net/download/jackandsnow/11210946) to download datasets.

2. Environment needed: Python 3, Paddlepaddle-gpu, NumPy, opencv, PIL, PyCharm(unnecessary)

3. In this project, firstly I use opencv to detect the main part of face in each picture, and then crop out the main part.
Secondly, I save all pictures' data with its label to **pickle file**, which is like *'[[data1, label1], [data2, label2], ...]'*.
The shape of data, the each value of which is in range(0, 1), is (14400,) and the label's type is integer range from 0 to 6.
Lastly, I set up a convolutional neural network by paddle.fluid, which contains two convolutional layers, two pooling llayers, one full connected layer and one softmax layer.

4. How to run: Firstly you just run 'python codes.preprocessing.py' (remember to replace the `image_dir` and `label_dir` with yours).
Then you can run 'python codes.main.py' to train the model and save the model. Finally you can load the model to predict your own face expressions.

5. About accuracy: As you can see in the following picture, I just run 10 epochs and the best accuracy can reach to 100%.
What's more, the model works well when doing prediction.
![accuracy.png](https://github.com/jackandsnow/FaceExpressionRecognition_PaddlePaddle/raw/master/resources/accuracy.png)
