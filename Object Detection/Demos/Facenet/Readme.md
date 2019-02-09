# :neckbeard: Face Recognition

### :books: Papers 

[FaceNet](http://arxiv.org/abs/1503.03832)

[DeepFace](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)

---

Code Credits: [david-sandberg](https://github.com/davidsandberg/facenet)

---

### Instructions

1. Add your custom data to `dataset/` **with each folder containing more than 1 image and proper name of each folder**.

2. Download any pretrained model from [here](https://github.com/davidsandberg/facenet#pre-trained-models) to `models/`. 

3. Run `facenet_train.py` and enjoy your :coffee: till then.

```
facenet_train.py 'TRAIN' dataset/ models/name_of_model.pb /path_and_name_of_new_classifier.pkl --batch_size=1 --image_size=160 --min_nrofimages_per_class=1
```

4. Edit `face.py` with your pretrained model name and classifier path and name.

```
facenet_model_checkpoint = os.path.dirname(__file__) + "/models/name_of_model.pb"
classifier_model = os.path.dirname(__file__) + "/path_and_name_of_new_classifier.pkl"
```

5. Run `real_time_face_recognition.py`. Cheers :beers:
