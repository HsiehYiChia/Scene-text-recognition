Scene text recognition
========
A real-time scene text recognition algorithm. Our system is able to recognize text in unconstrain background.  
This algorithm is based on [several papers](#references), and was implemented in C/C++.


Enviroment and dependency
-------- 
1. [OpenCV](http://opencv.org/) 3.1 or above
2. [CMake](https://cmake.org/) 3.10 or above
3. Visual Studio 2017 Community or above (Windows-only)


How to build?
--------
### Windows

1.  Install OpenCV; put the `opencv` directory into `C:\tools`
    -   You can install it manually from its Github repo, **or**
    -   You can install it via Chocolatey: `choco install opencv`, **or**
    -   If you already have OpenCV, edit `CMakeLists.txt` and change `WIN_OPENCV_CONFIG_PATH` to where you have it
2.  Use CMake to generate the project files
    ```bat
    cd Scene-text-recognition
    mkdir build-win
    cd build-win
    cmake .. -G "Visual Studio 15 2017 Win64"
    ```
3.  Use CMake to build the project
    ```bat
    cmake --build . --config Release
    ```
4.  Find the binaries in the root directory
    ```bat
    cd ..
    dir | findstr scene
    ```
5.  To execute the `scene_text_recognition.exe` binary, use its wrapper script; for example:
    ```bat
    .\scene_text_recognition.bat -i res\ICDAR2015_test\img_6.jpg
    ```

### Linux 

1.  Install OpenCV; refer to [OpenCV Installation in Linux](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
2.  Use CMake to generate the project files
    ```sh
    cd Scene-text-recognition
    mkdir build-linux
    cd build-linux
    cmake ..
    ```
3.  Use CMake to build the project
    ```sh
    cmake --build .
    ```
4.  Find the binaries in the root directory
    ```sh
    cd ..
    ls | grep scene
    ```
5.  To execute the binaries, run them as-is; for example:
    ```sh
    ./scene_text_recognition -i res/ICDAR2015_test/img_6.jpg
    ```

Usage
---------
The executable file `scene_text_recognition` must ultimately exist in the project root directory (i.e., next to `classifier/`, `dictionary/` etc.)
```
./scene_text_recognition -v:            take default webcam as input  
./scene_text_recognition -v [video]:    take a video as input  
./scene_text_recognition -i [image]:    take an image as input  
./scene_text_recognition -i [path]:     take folder with images as input,  
./scene_text_recognition -l [image]:    demonstrate "Linear Time MSER" Algorithm  
./scene_text_recognition -t detection:  train text detection classifier  
./scene_text_recognition -t ocr:        train text recognition(OCR) classifier 
```

Train your own classifier
---------
### Text detection
1. Put your text data to `res/pos`, non-text data to `res/neg`
2. Name your data in numerical, e.g. `1.jpg`, `2.jpg`, `3.jpg`, and so on.
3. Make sure `training` folder exist
4. Run `./scene_text_recognition -t detection`
```
mkdir training
./scene_text_recognition -t detection
```
5. Text detection classifier will be found at `training` folder

### Text recognition(OCR)
1. Put your training data to `res/ocr_training_data/` 
2. Arrange the data in `[Font Name]/[Font Type]/[Category]/[Character.jpg]`, for instance `Time_New_Roman/Bold/lower/a.jpg`. You can refer to `res/ocr_training_data.zip` 
3. Make sure `training` folder exist, and put `svm-train` to root folder (svm-train will be build by the system and should be found at build/)
4. Run `./scene_text_recognition -t ocr`
```
mkdir training
mv svm-train scene-text-recognition/
scene_text_recognition -t ocr
```
5. Text recognition(OCR) classifier will be fould at `training` folder


How it works
---------
The algorithm is based on an region detector called **Extremal Region (ER)**, which is basically the superset of famous region detector **MSER**. We use ER to find text candidates. The ER is extracted by **Linear-time MSER** algorithm. The pitfall of ER is repeating detection, therefore we remove most of repeating ERs with **non-maximum suppression**. We estimate the overlapped between ER based on the Component tree. and calculate the stability of every ER. Among the same group of overlapped ER, only the one with maximum stability is kept. After that we apply a 2-stages **Real-AdaBoost** to fliter non-text region. We choose **Mean-LBP** as feature because it's faster compare to other features. The suviving ERs are then group together to make the result from character-level to word level, which is more instinct for human. Our next step is to apply an OCR to these detected text. The chain-code of the ER is used as feature and the classifier is trained by **SVM**. We also introduce several post-process such as **optimal-path selection** and [spelling check](http://norvig.com/spell-correct.html) to make the recognition result better.  

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

#### Linear Time MSER Demo
The green pixels are so called **boundry pixels**, which are pushed into stacks. Each stack stand for a gray level, and pixels will be pushed according to their gary level. 
![result4](https://github.com/HsiehYiChia/canny_text/blob/master/res/demo_linear_time_MSER.gif)

References
----------
1. D. Nister and H. Stewenius, “Linear time maximally stable extremal regions,” European Conference on Computer Vision, pages 183196, 2008.
2. L. Neumann and J. Matas, “A method for text localization and recognition in real-world images,” Asian Conference on Computer Vision, pages 770783, 2010.
3. L. Neumann and J. Matas, “Real-time scene text localization and recognition,” Computer Vision and Pattern Recognition, pages 35383545, 2012.
4. L. Neumann and J. Matas, “On combining multiple segmentations in scene text recognition,” International Conference on Document Analysis and Recognition, pages 523527, 2013.
5. H. Cho, M. Sung and B. Jun, ”Canny Text Detector: Fast and robust scene text localization algorithm,” Computer Vision and Pattern Recognition, pages 35663573, 2016.
6. B. Epshtein, E. Ofek, and Y. Wexler, “Detecting text in natural scenes with stroke width transform,” Computer Vision and Pattern Recognition, pages 29632970, 2010.
7. P. Viola and M. J. Jones, “Rapid object detection using a boosted cascade of simple features,” Computer Vision and Pattern Recognition, pages 511518, 2001.
