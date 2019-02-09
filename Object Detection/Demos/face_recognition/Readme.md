# :skull: Face Recognition

## :pencil: Blog

Blog: [Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)

---

Code and Readme Credits: [ageitgey](https://github.com/ageitgey/face_recognition)

---

## Examples

#### Find faces in pictures

Find all the faces that appear in a picture:

![](https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png)

```python
import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_locations = face_recognition.face_locations(image)
```

#### Find and manipulate facial features in pictures

Get the locations and outlines of each person's eyes, nose, mouth and chin.

![](https://cloud.githubusercontent.com/assets/896692/23625282/7f2d79dc-025d-11e7-8728-d8924596f8fa.png)

```python
import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)
```

Finding facial features is super useful for lots of important stuff. But you can also use it for really stupid stuff
like applying [digital make-up](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py) (think 'Meitu'):

![](https://cloud.githubusercontent.com/assets/896692/23625283/80638760-025d-11e7-80a2-1d2779f7ccab.png)

#### Identify faces in pictures

Recognize who appears in each photo.

![](https://cloud.githubusercontent.com/assets/896692/23625229/45e049b6-025d-11e7-89cc-8a71cf89e713.png)

```python
import face_recognition
known_image = face_recognition.load_image_file("biden.jpg")
unknown_image = face_recognition.load_image_file("unknown.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
```

You can even use this library with other Python libraries to do real-time face recognition:

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py) for the code.

## Installation

### Requirements

  * Python 3.3+ or Python 2.7
  * macOS or Linux (Windows not officially supported, but might work)

### Installation Options:

#### Installing on Mac or Linux

First, make sure you have dlib already installed with Python bindings:

  * [How to install dlib from source on macOS or Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

Then, install this module from pypi using `pip3` (or `pip2` for Python 2):

```bash
pip3 install face_recognition
```

If you are having trouble with installation, you can also try out a
[pre-configured VM](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b).

#### Installing on Raspberry Pi 2+

  * [Raspberry Pi 2+ installation instructions](https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65)

#### Installing on Windows

While Windows isn't officially supported, helpful users have posted instructions on how to install this library:

  * [@masoudr's Windows 10 installation guide (dlib + face_recognition)](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508)

#### Installing a pre-configured Virtual Machine image

  * [Download the pre-configured VM image](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b) (for VMware Player or VirtualBox).


## Python Code Examples

All the examples are available [here](https://github.com/ageitgey/face_recognition/tree/master/examples).


#### Face Detection

* [Find faces in a photograph](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
* [Find faces in a photograph (using deep learning)](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)
* [Find faces in batches of images w/ GPU (using deep learning)](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py)
* [Blur all the faces in a live video using your webcam (Requires OpenCV to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/blur_faces_on_webcam.py)

#### Facial Features

* [Identify specific facial features in a photograph](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)
* [Apply (horribly ugly) digital make-up](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py)

#### Facial Recognition

* [Find and recognize unknown faces in a photograph based on photographs of known people](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)
* [Identify and draw boxes around each person in a photo](https://github.com/ageitgey/face_recognition/blob/master/examples/identify_and_draw_boxes_on_faces.py)
* [Compare faces by numeric face distance instead of only True/False matches](https://github.com/ageitgey/face_recognition/blob/master/examples/face_distance.py)
* [Recognize faces in live video using your webcam - Simple / Slower Version (Requires OpenCV to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py)
* [Recognize faces in live video using your webcam - Faster Version (Requires OpenCV to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py)
* [Recognize faces in a video file and write out new video file (Requires OpenCV to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_video_file.py)
* [Recognize faces on a Raspberry Pi w/ camera](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_on_raspberry_pi.py)
* [Run a web service to recognize faces via HTTP (Requires Flask to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/web_service_example.py)
* [Recognize faces with a K-nearest neighbors classifier](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py)
