
# coding: utf-8

# In[ ]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
import numpy as np
from scipy import misc
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
import math
from sklearn.cross_validation import KFold
from six import iteritems



parameters = []
conv_counter = 1
pool_counter = 1
affine_counter = 1

# In[ ]:

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_image_path_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file



def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        w, h, c = img.shape
        if (w != image_size) and (h != image_size):
            img = misc.imresize(img, (image_size, image_size, c))
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

def conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType, prefix, phase_train=True, use_batch_norm=True):
  global conv_counter
  global parameters
  name = prefix + '_' + str(conv_counter)
  conv_counter += 1
  with tf.name_scope(name) as scope:
    kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(inpOp, kernel, [1, dH, dW, 1], padding=padType)
    
    if use_batch_norm:
      conv_bn = batch_norm(conv, nOut, phase_train, 'batch_norm')
    else:
      conv_bn = conv
    biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv_bn, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  return conv1

def affine(inpOp, nIn, nOut):
  global affine_counter
  global parameters
  name = 'affine' + str(affine_counter)
  affine_counter += 1
  with tf.name_scope(name):
    kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                         trainable=True, name='biases')
    affine1 = tf.nn.relu_layer(inpOp, kernel, biases, name=name)
    parameters += [kernel, biases]
    return affine1
  
def lppool(inpOp, pnorm, kH, kW, dH, dW, padding):
  global pool_counter
  global parameters
  name = 'pool' + str(pool_counter)
  pool_counter += 1
  
  with tf.name_scope('lppool'):
    if pnorm == 2:
      pwr = tf.square(inpOp)
    else:
      pwr = tf.pow(inpOp, pnorm)
      
    subsamp = tf.nn.avg_pool(pwr,
                          ksize=[1, kH, kW, 1],
                          strides=[1, dH, dW, 1],
                          padding=padding,
                          name=name)
    subsamp_sum = tf.mul(subsamp, kH*kW)
    
    if pnorm == 2:
      out = tf.sqrt(subsamp_sum)
    else:
      out = tf.pow(subsamp_sum, 1/pnorm)
    
  return out

def mpool(inpOp, kH, kW, dH, dW, padding):
  global pool_counter
  global parameters
  name = 'pool' + str(pool_counter)
  pool_counter += 1
  with tf.name_scope('maxpool'):
    maxpool = tf.nn.max_pool(inpOp,
                   ksize=[1, kH, kW, 1],
                   strides=[1, dH, dW, 1],
                   padding=padding,
                   name=name)  
  return maxpool

def apool(inpOp, kH, kW, dH, dW, padding):
  global pool_counter
  global parameters
  name = 'pool' + str(pool_counter)
  pool_counter += 1
  return tf.nn.avg_pool(inpOp,
                        ksize=[1, kH, kW, 1],
                        strides=[1, dH, dW, 1],
                        padding=padding,
                        name=name)

def batch_norm(x, n_out, phase_train, name, affine=True):
  """
  Batch normalization on convolutional maps.
  Args:
      x:           Tensor, 4D BHWD input maps
      n_out:       integer, depth of input maps
      phase_train: boolean tf.Variable, true indicates training phase
      scope:       string, variable scope
      affine:      whether to affine-transform outputs
  Return:
      normed:      batch-normalized maps
  Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177
  """
  global parameters

  with tf.name_scope(name):

    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                       name=name+'/beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                        name=name+'/gamma', trainable=affine)
  
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = control_flow_ops.cond(phase_train,
                                      mean_var_with_update,
                                      lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
                                                        beta, gamma, 1e-3, affine, name=name)
    parameters += [beta, gamma]

  return normed


def inception(inp, inSize, ks, o1s, o2s1, o2s2, o3s1, o3s2, o4s1, o4s2, o4s3, poolType, name, phase_train=True, use_batch_norm=True):
  
  print('name = ', name)
  print('inputSize = ', inSize)
  print('kernelSize = {3,5}')
  print('kernelStride = {%d,%d}' % (ks,ks))
  print('outputSize = {%d,%d}' % (o2s2,o3s2))
  print('reduceSize = {%d,%d,%d,%d}' % (o2s1,o3s1,o4s2,o1s))
  print('pooling = {%s, %d, %d, %d, %d}' % (poolType, o4s1, o4s1, o4s3, o4s3))
  if (o4s2>0):
    o4 = o4s2
  else:
    o4 = inSize
  print('outputSize = ', o1s+o2s2+o3s2+o4)
  print()
  
  net = []
  
  with tf.name_scope(name):
    if o1s>0:
      conv1 = conv(inp, inSize, o1s, 1, 1, 1, 1, 'SAME', 'in1_conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm)
      net.append(conv1)
  
    if o2s1>0:
      conv3a = conv(inp, inSize, o2s1, 1, 1, 1, 1, 'SAME', 'in2_conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm)
      conv3 = conv(conv3a, o2s1, o2s2, 3, 3, ks, ks, 'SAME', 'in2_conv3x3', phase_train=phase_train, use_batch_norm=use_batch_norm)
      net.append(conv3)
  
    if o3s1>0:
      conv5a = conv(inp, inSize, o3s1, 1, 1, 1, 1, 'SAME', 'in3_conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm)
      conv5 = conv(conv5a, o3s1, o3s2, 5, 5, ks, ks, 'SAME', 'in3_conv5x5', phase_train=phase_train, use_batch_norm=use_batch_norm)
      net.append(conv5)
  
    if poolType=='MAX':
      pool = mpool(inp, o4s1, o4s1, o4s3, o4s3, 'SAME')
    elif poolType=='L2':
      pool = lppool(inp, 2, o4s1, o4s1, o4s3, o4s3, 'SAME')
    else:
      raise ValueError('Invalid pooling type "%s"' % poolType)
    
    if o4s2>0:
      pool_conv = conv(pool, inSize, o4s2, 1, 1, 1, 1, 'SAME', 'in4_conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm)
    else:
      pool_conv = pool
    net.append(pool_conv)
  
    incept = array_ops.concat(net, 3, name=name)
  return incept

def triplet_loss(anchor, positive, negative, alpha):
  """Calculate the triplet loss according to the FaceNet paper
  
  Args:
    anchor: the embeddings for the anchor images.
    positive: the embeddings for the positive images.
    positive: the embeddings for the negative images.
  Returns:
    the triplet loss according to the FaceNet paper as a float tensor.
  """
  with tf.name_scope('triplet_loss'):
    pos_dist = tf.reduce_sum(tf.square(tf.sub(anchor, positive)), 1)  # Summing over distances in each batch
    neg_dist = tf.reduce_sum(tf.square(tf.sub(anchor, negative)), 1)
    
    basic_loss = tf.add(tf.sub(pos_dist,neg_dist), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0, name='tripletloss')
    
  return loss

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay):
  """Setup training for the FaceNet model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    if optimizer=='ADAGRAD':
      opt = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer=='ADADELTA':
      opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif optimizer=='ADAM':
      opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    else:
      raise ValueError('Invalid optimization algorithm')

    grads = opt.compute_gradients(total_loss)
    
  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      moving_average_decay, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op, grads