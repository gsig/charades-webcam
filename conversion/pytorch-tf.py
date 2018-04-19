# @leonidk

from __future__ import print_function

import numpy as np 
import tensorflow as tf
import torch
import torch.nn as nn

import torchvision.models as models
import torchvision


def load_partial_state(model, state_dict):
    # @chenyuntc
    sd = model.state_dict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        if k not in sd or not sd[k].shape == v.shape:
            print('ignoring state key for loading: {}'.format(k))
            continue
        if isinstance(v, torch.nn.Parameter):
            v = v.data
        sd[k].copy_(v)


def change_output(model,nclass):
    if hasattr(model, 'classifier'):
        newcls = list(model.classifier.children())
        found = False
        for i, cls in reversed(list(enumerate(newcls))):
            if hasattr(cls, 'in_features'):
                newcls = newcls[:i] + [nn.Linear(cls.in_features, nclass)] + newcls[i + 1:]
                model.classifier = nn.Sequential(*newcls)
                found = True
                break
            if hasattr(cls, 'in_channels'):
                kwargs = {'kernel_size': (1, 1), 'stride': (1, 1)}
                newcls = newcls[:i] + [nn.Conv2d(cls.in_channels, nclass, **kwargs)] + newcls[i + 1:]
                model.classifier = nn.Sequential(*newcls)
                if hasattr(model, 'num_classes'):
                    model.num_classes = nclass
                found = True
                break
        assert found


model = models.squeezenet1_1(pretrained=True)
state = torch.load('/nfs.yoda/gsigurds/ai2/caches/rgbnet_squeezenet1/model_best.pth.tar')['state_dict']
change_output(model, 157) 
load_partial_state(model, state)
destination_py = 'squeezenet.py'

type_lookups = {}
outfp = open(destination_py,'w')
outfp.write('import tensorflow as tf\n\n')
out_s = ''
def conv2d(c,**kwargs):
    padding = 'VALID' if c.padding[0] is 0 else 'SAME'
    filters = c.out_channels
    size = c.kernel_size
    parameters = [p for p in c.parameters()]
    W = parameters[0].data.numpy()
    if len(parameters) > 1:
        b = parameters[1].data.numpy()

    W = np.transpose(W,[2,3,1,0])

    wi = tf.constant_initializer(W)
    if len(parameters) > 1:
        bi = tf.constant_initializer(b)
    Wt = tf.get_variable('weights',shape=W.shape,initializer=wi)#,
    if 'print' not in kwargs or kwargs['print'] == True:
        outfp.write(out_s + 'W = tf.get_variable("weights",shape=[{},{},{},{}])\n'.format(*list(W.shape)))

    if len(parameters) > 1:
        bt = tf.get_variable('bias',shape=b.shape,initializer=bi)#,
        if 'print' not in kwargs or kwargs['print'] == True:
            outfp.write(out_s + 'b = tf.get_variable("bias",shape=[{}])\n'.format(b.shape[0]))
    x = tf.nn.conv2d(kwargs['inp'],Wt,[1,c.stride[0],c.stride[1],1],padding)
    if 'print' not in kwargs or kwargs['print'] == True:
        outfp.write(out_s + 'x = tf.nn.conv2d(x,W,[1,{},{},1],"{}")\n'.format(c.stride[0],c.stride[1],padding))
    if len(parameters) > 1:
        x = tf.nn.bias_add(x,bt)
        if 'print' not in kwargs or kwargs['print'] == True:
            outfp.write(out_s + 'x = tf.nn.bias_add(x,b)\n')

    return x

def relu(c,**kwargs):
    outfp.write(out_s + "x = tf.nn.relu(x)\n")
    return tf.nn.relu(kwargs['inp'])
def max_pool(c,**kwargs):
    padding = 'VALID' if c.padding is 0 else 'SAME'
    outfp.write(out_s + "x = tf.nn.max_pool(x,[1,{0},{0},1],strides=[1,{1},{1},1],padding='{2}')\n".format(
        c.kernel_size,c.stride,padding))
    x = tf.nn.max_pool(kwargs['inp'],[1,c.kernel_size,c.kernel_size,1],strides=[1,c.stride,c.stride,1],padding=padding)
    return x
def avg_pool(c,**kwargs):
    padding = 'VALID' if c.padding is 0 else 'SAME'
    outfp.write(out_s + "x = tf.nn.avg_pool(x,[1,{0},{0},1],strides=[1,{1},{1},1],padding='{2}')\n".format(
        c.kernel_size,c.stride,padding))
    x = tf.nn.avg_pool(kwargs['inp'],[1,c.kernel_size,c.kernel_size,1],strides=[1,c.stride,c.stride,1],padding=padding)
    return x
def dropout(c,**kwargs):
    outfp.write(out_s + 'x = x\n')
    return kwargs['inp']
def fire_module(c,**kwargs):
    global out_s

    # couldn't figure out how to
    # automatically unravel it
    outfp.write(out_s + "x = fire_module(x,{0},{1},{2},{3})\n".format(
        c.squeeze.in_channels,c.squeeze.out_channels,c.expand1x1.out_channels,c.expand3x3.out_channels))
    with tf.variable_scope("fire"):
        with tf.variable_scope("squeeze"):
            s = conv2d(c.squeeze,inp=kwargs['inp'],print=False)
            s = tf.nn.relu(s)
        with tf.variable_scope("e11"):
            e11 = conv2d(c.expand1x1,inp=s,print=False)
            e11 = tf.nn.relu(e11)
        with tf.variable_scope("e33"):
            e33 = conv2d(c.expand3x3,inp=s,print=False)
            e33 = tf.nn.relu(e33)
    x = tf.concat([e11,e33],3)
    return x

def seq_container(c,**kwargs):
    global out_s
    x = kwargs['inp']
    for c2 in enumerate(c.children()):
        c2_class = c2[1].__class__
        if c2_class in type_lookups:
            outfp.write(out_s + "with tf.variable_scope('{}'):\n".format('layer' + str(c2[0])))
            with tf.variable_scope('layer' + str(c2[0])):
                out_s = out_s + '    '
                x = type_lookups[c2_class](c2[1],inp = x)
                name = kwargs['name'] if 'name' in kwargs else ''
                outfp.write(out_s + "self.layers.append(x)\n".format(name + str(c2[0])))

                out_s = out_s[:-4]
        else:
            unknown_class(c2[1])
            print(c2_class)
    return x
def batch_norm(c,**kwargs):
    print('batch_norm')
    return kwargs['inp']
type_lookups[torch.nn.modules.conv.Conv2d] = conv2d
type_lookups[torch.nn.modules.activation.ReLU] = relu
type_lookups[torch.nn.modules.container.Sequential] = seq_container
type_lookups[torch.nn.modules.pooling.MaxPool2d] = max_pool
type_lookups[torch.nn.modules.pooling.AvgPool2d] = avg_pool
type_lookups[torch.nn.modules.dropout.Dropout] = dropout
type_lookups[torchvision.models.squeezenet.Fire] = fire_module
type_lookups[torch.nn.modules.batchnorm.BatchNorm2d] = batch_norm
tf.reset_default_graph()
input_image = tf.placeholder('float',shape=[None,None,None,3],name='input_image')

if True:
    outfp.write('def fire_module(x,inp,sp,e11p,e33p):\n')
    outfp.write('    with tf.variable_scope("fire"):\n')
    outfp.write('        with tf.variable_scope("squeeze"):\n')
    outfp.write('            W = tf.get_variable("weights",shape=[1,1,inp,sp])\n')
    outfp.write('            b = tf.get_variable("bias",shape=[sp])\n')
    outfp.write('            s = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")+b\n')
    outfp.write('            s = tf.nn.relu(s)\n')
    outfp.write('        with tf.variable_scope("e11"):\n')
    outfp.write('            W = tf.get_variable("weights",shape=[1,1,sp,e11p])\n')
    outfp.write('            b = tf.get_variable("bias",shape=[e11p])\n')
    outfp.write('            e11 = tf.nn.conv2d(s,W,[1,1,1,1],"VALID")+b\n')
    outfp.write('            e11 = tf.nn.relu(e11)\n')
    outfp.write('        with tf.variable_scope("e33"):\n')
    outfp.write('            W = tf.get_variable("weights",shape=[3,3,sp,e33p])\n')
    outfp.write('            b = tf.get_variable("bias",shape=[e33p])\n')
    outfp.write('            e33 = tf.nn.conv2d(s,W,[1,1,1,1],"SAME")+b\n')
    outfp.write('            e33 = tf.nn.relu(e33)\n')
    outfp.write('        return tf.concat([e11,e33],3) \n\n')


if len([_ for _ in model.children()]) == 2:
    outfp.write('class SqueezeNet:\n')
    out_s += '    '
    outfp.write(out_s + 'def __init__(self):\n')
    
    for idx,c in enumerate(model.children()):
        out_s = out_s + '    '

        if idx is 0:
            outfp.write(out_s+"self.image = tf.placeholder('float',shape=[None,None,None,3],name='input_image')\n")
            outfp.write(out_s+"self.layers = []\n")

            outfp.write(out_s+'x = self.image\n')
            outfp.write(out_s+"with tf.variable_scope('features'):\n")
            with tf.variable_scope('features'):
                out_s = out_s + '    '
                features = type_lookups[c.__class__](c,inp=input_image)
                out_s = out_s[:-4]

            outfp.write(out_s+'self.features = x\n')

        elif idx is 1:
            outfp.write(out_s+"with tf.variable_scope('classifier'):\n")
            with tf.variable_scope('classifier'):
                out_s = out_s + '    '
                classifier = type_lookups[c.__class__](c,inp=features)
                #classifier = tf.reshape(classifier,[-1,1000])
                classifier = tf.reshape(classifier,[-1,157])
                out_s = out_s[:-4]

            #outfp.write(out_s+'self.classifier = tf.reshape(x,[-1,1000])\n')
            outfp.write(out_s+'self.classifier = tf.reshape(x,[-1,157])\n')
            outfp.write('\n\n')
        out_s = out_s[:-4]


else:
    x = input_image
    for idx,c in enumerate(model.children()):
        x = type_lookups[c.__class__](c,inp=x)
outfp.close()


print(classifier.get_shape(),classifier.name,input_image.name,features.name)



from PIL import Image
from scipy.misc import imresize
import os

with open('labels.txt') as fp:
    labels = [c[:-2].split(':')[1] for c in fp.readlines()]
def get_img(filename):
    vec = np.array(Image.open(filename))
    vec = imresize(vec,(224,224)).astype(np.float32)/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    vec = (vec-mean)/std
    return vec
    
img_dir = '.'
img_names = [x for x in os.listdir(img_dir) if 'jpeg' in x.lower()]
imgs = [get_img(os.path.join(img_dir,x)) for x in img_names]

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
scores = sess.run(classifier,feed_dict={input_image:np.array(imgs).reshape([-1,224,224,3])})
for idx,s in enumerate(np.argmax(scores,1)):
    print(img_names[idx],labels[s])




saver.save(sess, 'squeezenet.ckpt')


from torch.autograd import Variable
input_data = torch.FloatTensor(np.transpose(np.array(imgs),[0,3,1,2]))
model.eval()
pyt_scores = model(Variable(input_data))
scores_ref = pyt_scores.data.numpy()


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
print(rel_error(scores,scores_ref))
