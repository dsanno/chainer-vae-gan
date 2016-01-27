import argparse
import numpy as np
import io
import os
from PIL import Image
import cPickle as pickle

import chainer
from chainer import cuda, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import net

parser = argparse.ArgumentParser(description='DCGAN trainer for ETL9')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', required=True, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output image file path without extension')
parser.add_argument('--dataset', '-d', default='dataset/images.pkl', type=str,
                    help='dataset file path')
parser.add_argument('--size', '-s', default=64, type=int, choices=[48, 64, 80, 96, 112, 128],
                    help='image size')
args = parser.parse_args()

image_size = args.size
enc = net.Encoder(density=4, size=image_size)
gen = net.Generator(density=4, size=image_size)

serializers.load_hdf5(args.input + '.enc.model', enc)
serializers.load_hdf5(args.input + '.gen.model', gen)

with open(args.dataset, 'rb') as f:
    images = pickle.load(f)

gpu_device = None
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu

latent_size = gen.latent_size
out_image_row_num = 10
out_image_col_num = 10

if gpu_device == None:
    enc.to_cpu()
    gen.to_cpu()
    xp = np
else:
    enc.to_gpu(gpu_device)
    gen.to_gpu(gpu_device)
    xp = cuda.cupy

x = np.zeros((out_image_row_num * 2, 3, image_size, image_size), dtype=np.float32)
perm = np.random.permutation(len(images))
for j, p in enumerate(perm[:out_image_row_num]):
    image = images[p]
    offset_x = np.random.randint(8) + 13
    offset_y = np.random.randint(8) + 33
    w = 144
    h = 144
    with io.BytesIO(image) as b:
        pixels = np.asarray(Image.open(b).convert('RGB').crop((offset_x, offset_y, offset_x + w, offset_y + h)).resize((image_size, image_size)))
        pixels = pixels.astype(np.float32).transpose((2, 0, 1)).reshape((3, image_size, image_size))
        x[j * 2] = pixels / 127.5 - 1
        x[j * 2 + 1] = pixels[:,:,::-1] / 127.5 - 1
z0, mean, var = enc(Variable(xp.asarray(x), volatile=True), train=False)
z = xp.zeros((out_image_row_num * out_image_col_num, latent_size)).astype(np.float32)
for j in range(out_image_row_num):
    for k in range(1, out_image_col_num):
        w = np.float32((k - 1)) / (out_image_col_num - 2)
        z[j * out_image_col_num + k] = z0.data[j * 2] * (1 - w) + z0.data[j * 2 + 1] * w
y = gen(Variable(z, volatile=True), train=False)
image = cuda.to_cpu(y.data)
for j in range(out_image_row_num):
    image[j * out_image_col_num] = x[j * 2]
image = ((image + 1) * 128).clip(0, 255).astype(np.uint8)
image = image.reshape((out_image_row_num, out_image_col_num, 3, image_size, image_size)).transpose((0, 3, 1, 4, 2)).reshape((out_image_row_num * image_size, out_image_col_num * image_size, 3))
Image.fromarray(image).save(args.output)
