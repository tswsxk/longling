
# coding: utf-8

# In[4]:

import mxnet as mx
import numpy as np
import random
import urllib2
from StringIO import StringIO
from skimage import io,transform,color
from collections import namedtuple
import matplotlib.pyplot as plt
Batch = namedtuple('Batch', ['data'])


# In[5]:

from skimage.transform import resize
import matplotlib.cm as cm

def get(u):
    req = urllib2.Request(u)
    res = urllib2.urlopen(req)
    x = res.read()
    return x

def _conv(location):
    img = io.imread(location)

    if len(img.shape) == 2:
        img = color.grey2rgb(img) 
    elif img.shape[-1] == 4:
        img = color.rgba2rgb(img)
    elif img.shape[-1] == 3 and len(img.shape) == 4: 
        img = random.choice(img)  
    elif len(img.shape) == 3 and img.shape[-1] > 4:  
        img = random.choice(img)
        img = color.grey2rgb(img)
        
    img = transform.resize(img, (224, 224)) * 255
    img = img.transpose((2,0,1))
    img = img[np.newaxis, :]
    return img

def get_image(u):
    img = get(u)
    img = StringIO(img)
    img = _conv(img)
    return img

def get_image_batch(u):
    img = get(u)
    img = StringIO(img)
    img = _conv(img)
    return Batch([mx.nd.array(img)])

def generate_cam(output, weights):
    cam = np.zeros((7,7), dtype = np.float32)	# [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[i, :, :]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-5)
    cam = cam / np.max(cam)
    cam = resize(cam, (224,224))
    cam = cm.jet(cam,alpha=0.5)
    cam = cam[: ,:, 0:3]
    cam /= np.max(cam)

    return cam


# In[6]:

sym, arg_params, aux_params = mx.model.load_checkpoint('model/cnn', 5)

#mod to generate conv
all_layers = sym.get_internals()
sym_plus = all_layers['_plus32_output']
sym_conv = all_layers['stage4_unit3_conv3_output']
group = mx.sym.Group([sym_plus, sym_conv])
mod_conv = mx.mod.Module(symbol=group, data_names=('data',),  context=mx.gpu(0)) # !!!
mod_conv.bind(for_training = False, data_shapes=[['data', (1L, 3L, 224L, 224L)]])
mod_conv.set_params(arg_params, aux_params, allow_missing=False)


# mod to get grad
sym, arg_params, aux_params = mx.model.load_checkpoint('model/cnn', 3)
num_classes = 2
data = mx.sym.Variable(name='data')
bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=0.9, name='bn1')
relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
flat = mx.symbol.Flatten(data=pool1)
fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='full1')
loss = mx.symbol.max(fc1, axis=1)
net = mx.symbol.MakeLoss(loss)
#net = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')


mod_grad = mx.mod.Module(symbol=net, data_names=('data',), context=mx.gpu(0)) # !!!
mod_grad.bind(for_training = False, data_shapes=[['data', (1L, 2048L, 7L, 7L)]], inputs_need_grad=False)
mod_grad.set_params(arg_params, aux_params, allow_missing=False, force_init=True)


# In[82]:

mod_conv.forward(batch,is_train=False)
out_plus, out_conv = mod_conv.get_outputs()


# In[83]:

#layer = mx.nd.array(np.random.randn(1,2048,7,7))
import time
st = time.time()
mod_grad.forward(Batch([out_plus]), is_train=False)

print mod_grad.get_outputs()[0].asnumpy()
print time.time() - st


# In[7]:

def eval_numerical_gradient(f, x, verbose=True, h=1e-3):
    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    cnt = 0
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore
        #print fxph, fxmh
        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        cnt += 1
        if verbose and cnt % 500 == 0:
            print ix, grad[ix]
        
        it.iternext() # step to next dimension

    return grad

def eval_numerical_gradient_array(f, x, df = 1, h=1e-3):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    cnt = 0
    while not it.finished:
        ix = it.multi_index
    
        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval
    
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        cnt += 1 
        if cnt % 500 == 0:
            print ix, grad[ix]
        it.iternext()
    return grad

def get_output(x, mod):
    mod.forward(Batch([mx.nd.array(x) ]), is_train=False)
    out = mod.get_outputs()[0].asnumpy()
    out = out[0]
    return out


# In[85]:

#f = lambda x: get_output(x, mod_grad)
grad = eval_numerical_gradient_array(get_output, out_plus.asnumpy())
save = grad.copy()


# In[86]:

grad = grad[0]
grad = grad / ( np.sqrt( np.sum(grad * grad) ) +  1e-5)
weights = np.mean(grad, axis = (1, 2)) 			# [2048]
print weights.shape

output = out_conv.asnumpy()[0]
print output.shape


# In[13]:

#url = 'http://i.pstatp.com/origin/1b64003690c9ffa023ce'
#url = 'http://i.pstatp.com/origin/23e2002e8d2345b6a349'
url = 'http://happytagger.byted.org/image/9742d1590d0296984545e1907b1bbb17.jpg'
batch = get_image_batch(url)
img = get_image(url)

mod_conv.forward(batch,is_train=False)
out_plus, out_conv = mod_conv.get_outputs()

grad = eval_numerical_gradient_array(lambda x: get_output(x, mod_grad), out_plus.asnumpy())
save = grad.copy()

grad = grad[0]
grad = grad / ( np.sqrt( np.sum(grad * grad) ) +  1e-5)
weights = np.mean(grad, axis = (1, 2)) 			# [2048]
print weights.shape

output = out_conv.asnumpy()[0]
print output.shape

cam = generate_cam(output, weights)
img = get_image(url)[0]
img = np.transpose(img, (1,2,0) )
img /= img.max()



new_img = img+3*cam
new_img /= new_img.max()

   


# In[14]:



plt.figure(figsize=(10, 8))

plt.subplot(211)
plt.axis('off')
plt.imshow(img)
plt.subplot(212)
plt.axis('off')
plt.imshow(new_img)


# In[12]:

batch = get_image_batch('http://happytagger.byted.org/image/9742d1590d0296984545e1907b1bbb17.jpg')
sym, arg_params, aux_params = mx.model.load_checkpoint('./res-101', 3)


mod = mx.mod.Module(symbol=sym, data_names=('data',), label_names=('softmax1_label',), context=mx.cpu()) # !!!
mod.bind(for_training=False, data_shapes=[['data', (1L, 3L, 224L, 224L)]], )
mod.set_params(arg_params, aux_params, allow_missing=False)
mod.forward(batch)
z = mod.get_outputs()[0].asnumpy()
print z.shape
print z


# In[ ]:



