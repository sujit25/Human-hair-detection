# Human Hair Detection

Java based GUI application for extraction of human hair region from input image of human face using OpenCV.

[![Build Status](https://travis-ci.org/Promact/md2.svg?branch=master)](https://travis-ci.org/Promact/md2)

### Prerequisities
* Oracle Java v8.0 (64bit)
* JavaFX
* Netbeans >= v8.0
* OpenCV 3.1.0 native installation


### Getting Started
##### Install Java with NetBeans
* Visit [oracle](http://www.oracle.com/technetwork/java/javase/downloads/jdk-netbeans-jsp-142931.html) and download JDK 8 with Netbeans for your platform. JDK 8 comes bundled with JavaFX, so you don't have to install JDK manually.
![Download JDK with NetBeans](https://i.imgur.com/c1wKQv1.png)
##### Install OpenCV
* Visit [openCV](https://github.com/opencv/opencv/releases/tag/3.1.0) and download the source code.
![Download OpenCV](https://i.imgur.com/ipd7DK6.png)
* Extract the contents of ZIP in a directory and follow the instructions in the link below to compile OpenCV natively.
[http://opencv-java-tutorials.readthedocs.io/en/latest/01-installing-opencv-for-java.html](http://opencv-java-tutorials.readthedocs.io/en/latest/01-installing-opencv-for-java.html)


##### Setting up the project in Net Beans
* Launch NetBeans
![Launch Net Beans](https://i.imgur.com/U7sW2N8.png)

* Clone this project in your local machine
#### HTTPS
```
git clone https://github.com/sujit25/Human-hair-detection.git
```
#### SSH
```
git clone git@github.com:sujit25/Human-hair-detection.git
```

* Add reference to opencv-310.jar available from binaries generated during compilation.

![Add reference to opencv-310.jar](https://i.imgur.com/2Yiw8SF.png)
    
* Go to Hair Detection Netbeans Project > Properties > Run > VM Options : add 
```
-Djava.library.path=/path/to/lib/containing/opencv_libraries
```
for e.g. on my machine: 
```
-Djava.library.path="/home/sujit25/softwares/opencv-3.1.0-with-cuda8/release/lib"
```

### Note: I have not tested application with any OpenCV version above 3.1.0.

##### Running the Project
* When you launch the project, you will see a window where you can choose source image.
![Empty Window](https://i.imgur.com/t6vEE8C.png) 

* Choose a source image from your local directory. You can select one from the examples provided in this repository.
![Source image window](https://i.imgur.com/FU0JXyb.png)

* Click the button to generate result images.
![Generated image window](https://i.imgur.com/NdEMZtd.png)

* All the generated images are saved in your local directory.
![Generated output](https://i.imgur.com/xxHs3AR.png)
