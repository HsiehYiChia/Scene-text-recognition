Scene text recognition
========
A real-time scene text recognition algorithm. Our system is able to recognize text in unconstrain background.  
This algorithm is based on [several papers](#references), and was implemented in C/C++.


Enviroment and dependency
--------
1. Microsoft Windows 10
2. Visual Studio 2015 Community or above
3. [OpenCV](http://opencv.org/) 3.1 or above
4. [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) 3.21 or above
5. [ICDAR](http://u-pat.org/ICDAR2017/) dataset


How to?
--------
### Via Visual Studio:
1. Put the `opencv` directory to `C:\` 
2. Open the `*.vcxproj` project file(or you can setup a OpenCV project for Visual Studio by yourself). It's recommend to enable the OpenMP flag to speed up system performance.  
3. Config input arguments 
  `Properties Pages -> Coniguration Properties -> Debugging -> Command Arguments`  
4. Press `Ctrl+F5` to execute and `Esc` to end.  
  
### Via Command Prompt:  
1. Compile the `.exe` via Visual Studio and you'll find `.exe` in `x64/Release/`  
2. Put `.exe`, `opencv_world3xx.dll`, `er_classifier/`, `ocr_classifier/`, `dictionary/` in the same directory.  
3. Usage:  
`*.exe -v`: take default webcam as input
`*.exe -v [infile]`: take video as input
`*.exe -i [infile]`: take image as input  
`*.exe -icdar`: take icdar dataset as input  


How it works
---------
The algorithm is based on an region detector called **Extremal Region (ER)**, which is basically the superset of famous region detector **MSER**. We use ER to find text candidates. The ER is extracted by **Linear-time MSER** algorithm. The pitfall of ER is repeating detection, therefore we remove most of repeating ERs with **non-maximum suppression**. We estimate the overlapped between ER based on the Component tree. and calculate the stability of every ER. Among the same group of overlapped ER, only the one with maximum stability is kept. After that we apply a 2-stages **Real-AdaBoost** to fliter non-text region. We choose **Mean-LBP** as feature because it's faster compare to other features. The suviving ERs are then group together to make the result from character-level to word level, which is more instinct for human. Our next step is to apply an OCR to these detected text. The chain-code of the ER is used as feature and the classifier is trained by **SVM**. We also introduce several post-process such as `optimal-path selection` and [spelling check](http://norvig.com/spell-correct.html) to make the recognition result better.  

![overview](https://github.com/HsiehYiChia/canny_text/blob/master/res/overview.jpg)


Notes
---------
For text classification, the training data contains 12,000 positive samples, mostly extract from ICDAR 2003 and ICDAR 2015 dataset. the negative sample are extracted from random images with a bootstrap process. As for OCR classification, the training data is consist of purely synthetic letters, including 28 different fonts.  

The system is able to detect text in real-time(30FPS) and recognize text in nearly real-time(8~15 FPS, depends on number of texts) for a 640x480 resolution image on a Intel Core i7 desktop computer. The algorithm's end-to-end text detection accuracy on ICDAR dataset 2015 is roughly 70% with fine tune, and end-to-end recognition accuracy is about 30%.


Result
----------
#### Detection result on IDCAR 2015  
![result1](https://github.com/HsiehYiChia/canny_text/blob/master/res/reuslt1.jpg)
![result2](https://github.com/HsiehYiChia/canny_text/blob/master/res/reuslt2.jpg)
![result3](https://github.com/HsiehYiChia/canny_text/blob/master/res/reuslt3.jpg)

#### Recognition result on random image 
![result4](https://github.com/HsiehYiChia/canny_text/blob/master/res/reuslt4.jpg)
![result5](https://github.com/HsiehYiChia/canny_text/blob/master/res/reuslt5.jpg)


References
----------
1. D. Nister and H. Stewenius, “Linear time maximally stable extremal regions,” European Conference on Computer Vision, pages 183–196, 2008.
2. L. Neumann and J. Matas, “A method for text localization and recognition in real-world images,” Asian Conference on Computer Vision, pages 770–783, 2010.
3. L. Neumann and J. Matas, “Real-time scene text localization and recognition,” Computer Vision and Pattern Recognition, pages 3538–3545, 2012.
4. L. Neumann and J. Matas, “On combining multiple segmentations in scene text recognition,” International Conference on Document Analysis and Recognition, pages 523–527, 2013.
5. H. Cho, M. Sung and B. Jun, ”Canny Text Detector: Fast and robust scene text localization algorithm,” Computer Vision and Pattern Recognition, pages 3566–3573, 2016.
6. B. Epshtein, E. Ofek, and Y. Wexler, “Detecting text in natural scenes with stroke width transform,” Computer Vision and Pattern Recognition, pages 2963–2970, 2010.
7. P. Viola and M. J. Jones, “Rapid object detection using a boosted cascade of simple features,” Computer Vision and Pattern Recognition, pages 511–518, 2001.
