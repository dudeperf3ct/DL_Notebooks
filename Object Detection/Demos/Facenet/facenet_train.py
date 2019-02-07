
# coding: utf-8

# In[2]:

import os
import sys
import argparse
import numpy as np
import keras
import tensorflow as tf
import facenet
import math
import pickle
from sklearn.svm import SVC

# In[ ]:

def main(args):
    
    with tf.Graph().as_default():
        
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)

            dataset = facenet.get_dataset(args.data_dir)

            for cls in dataset:
                assert len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset'

            paths, labels = facenet.get_image_path_and_labels(dataset)

            print ('Number of Classes: %d' %len(dataset))
            print ('Number of Images: %d' %len(paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
            
                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy) 


# In[ ]:

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'], 
                        help='Indicates if a new classifier should be trained or a classification ' + 
                        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str, help='Path to the data directory containing images')
    parser.add_argument('model', type=str, 
                help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', type=str,
                       help='Classifier model file name as a pickle (.pkl) file. ' + 
                        'For training this is the output and for classification this is an input.')
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batches', default=1)
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels', default=160)
    parser.add_argument('--seed', type=int, help='Random Seed', default=42)
    parser.add_argument('--min_nrofimages_per_class', type=int, 
                        help='Include only classes having this minimum number of images', default=1)
    parser.add_argument('--nrof_train_images_per_class', type=int, 
                        help='Use this number of images from each class for trainig and rest of testing', default=1)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

