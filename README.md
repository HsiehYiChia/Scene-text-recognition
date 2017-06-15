Scene text recognition
========

A real-time scene text recognition algorithm.
Our system is able to recognize text in unconstrain background.
This algorithm is based on several papers, and was implemented in C/C++.


Enviroment and dependency
--------

1. Microsoft Windows 10
2. Visual Studio 2015 Community
3. [OpenCV][1] 3.1 or above
4. [libsvm][2] 3.22 or above
5. [ICDAR][3] dataset


Setup
--------

Put the `opencv` directory to `C:\` and setup a OpenCV project for Visual Studio. It's recommend to enable the OpenMP flag to speed up the system performance.


How it works
---------

The algorithm is based on an region detector called **Extremal Region (ER)**, which is basically the superset of famous region detector MSER. We use ER to find text candidates. The ER is extracted by **Linear-time MSER** algorithm. The pitfall of ER is repeating detection, therefore we remove most of repeating ERs with non-maximum suppression. We estimate the overlapped between ER based on the Component tree. and calculate the stability of every ER. Among the same group of overlapped ER, only the one with maximum stability is kept. After that we apply a 2-stages **Real-AdaBoost** to fliter non-text region. We choose **Mean-LBP** as feature because it's faster compare to other features. The suviving ERs are then group together to make the result from character-level to word level, which is more instinct for human. Our next step is to apply an OCR to these detected text. The chain-code of the ER is used as feature and the classifier is trained by **SVM**. We also introduce several post-process such as optimal-path selection and [spelling check][4] to make the recognition result better.  

![overview](https://github.com/HsiehYiChia/canny_text/blob/master/res/overview.jpg)


Notes
---------

For text classification, the training data contains 12,000 positive samples, mostly extract from ICDAR 2003 and ICDAR 2015 dataset. the negative sample are extracted from random images with a bootstrap process. As for OCR classification, the training data is consist of purely synthetic letters, including 28 different fonts.  

The system is able to detect text in real-time(30FPS) and recognize text in nearly real-time(8~15 FPS, depends on number of texts) for a 640x480 resolution image on a standard desktop computer.


Result
----------

#### Detection result on IDCAR 2015  
![result1](https://github.com/HsiehYiChia/canny_text/blob/master/res/reuslt1.jpg)
![result2](https://github.com/HsiehYiChia/canny_text/blob/master/res/reuslt2.jpg)
![result3](https://github.com/HsiehYiChia/canny_text/blob/master/res/reuslt3.jpg)

#### Recognition result on random image 
![result4](https://github.com/HsiehYiChia/canny_text/blob/master/res/reuslt4.jpg)
![result5](https://github.com/HsiehYiChia/canny_text/blob/master/res/reuslt5.jpg)



[1]: http://opencv.org/
[2]: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
[3]: http://u-pat.org/ICDAR2017/
[4]: http://norvig.com/spell-correct.html
