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

parser = argparse.ArgumentParser(description='VAE and DCGAN trainer')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output model file path without extension')
parser.add_argument('--iter', default=100, type=int,
                    help='number of iteration')
parser.add_argument('--out_image_dir', default=None, type=str,
                    help='output directory to output images')
parser.add_argument('--dataset', '-d', default='dataset/images.pkl', type=str,
                    help='dataset file path')
parser.add_argument('--size', '-s', default=64, type=int, choices=[48, 64, 80, 96, 112, 128],
                    help='image size')
args = parser.parse_args()

image_size = args.size
enc_model = net.Encoder(density=4, size=image_size)
gen_model = net.Generator(density=4, size=image_size)
dis_model = net.Discriminator(density=4, size=image_size)

optimizer_enc = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_enc.setup(enc_model)
optimizer_enc.add_hook(chainer.optimizer.WeightDecay(0.00001))
optimizer_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_gen.setup(gen_model)
optimizer_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
optimizer_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_dis.setup(dis_model)
optimizer_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

if args.input != None:
    serializers.load_hdf5(args.input + '.enc.model', enc_model)
    serializers.load_hdf5(args.input + '.enc.state', optimizer_enc)
    serializers.load_hdf5(args.input + '.gen.model', gen_model)
    serializers.load_hdf5(args.input + '.gen.state', optimizer_gen)
    serializers.load_hdf5(args.input + '.dis.model', dis_model)
    serializers.load_hdf5(args.input + '.dis.state', optimizer_dis)

if args.out_image_dir != None:
    if not os.path.exists(args.out_image_dir):
        try:
            os.mkdir(args.out_image_dir)
        except:
            print 'cannot make directory {}'.format(args.out_image_dir)
            exit()
    elif not os.path.isdir(args.out_image_dir):
        print 'file path {} exists but is not directory'.format(args.out_image_dir)
        exit()

with open(args.dataset, 'rb') as f:
    images = pickle.load(f)

gpu_device = None
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu

latent_size = gen_model.latent_size
BATCH_SIZE = 100
image_save_interval = 10000

def train_one(enc, gen, dis, optimizer_enc, optimizer_gen, optimizer_dis, x_batch, gpu_device):
    batch_size = len(x_batch)
    if gpu_device == None:
        xp = np
    else:
        xp = cuda.cupy
    # encode
    x_in = xp.asarray(x_batch)
    z0, mean, var = enc(Variable(x_in))
    x0 = gen(z0)
    y0, l0 = dis(x0)
    loss_enc = F.gaussian_kl_divergence(mean, var) / float(l0.data.size)
    loss_gen = 0
    loss_gen = F.softmax_cross_entropy(y0, Variable(xp.zeros(batch_size).astype(np.int32)))
    loss_dis = F.softmax_cross_entropy(y0, Variable(xp.ones(batch_size).astype(np.int32)))
    # train generator
    z1 = Variable(xp.random.normal(0, 1, (batch_size, latent_size)).astype(np.float32))
    x1 = gen(z1)
    y1, l1 = dis(x1)
    loss_gen += F.softmax_cross_entropy(y1, Variable(xp.zeros(batch_size).astype(np.int32)))
    loss_dis += F.softmax_cross_entropy(y1, Variable(xp.ones(batch_size).astype(np.int32)))
    # train discriminator
    y2, l2 = dis(Variable(xp.asarray(x_batch)))
    loss_enc += F.mean_squared_error(l0, l2)
    loss_gen += 0.1 * F.mean_squared_error(l0, l2)
    loss_dis += F.softmax_cross_entropy(y2, Variable(xp.zeros(batch_size).astype(np.int32)))

    optimizer_enc.zero_grads()
    loss_enc.backward()
    optimizer_enc.update()

    optimizer_gen.zero_grads()
    loss_gen.backward()
    optimizer_gen.update()

    optimizer_dis.zero_grads()
    loss_dis.backward()
    optimizer_dis.update()

    return (float(loss_enc.data), float(loss_gen.data), float(loss_dis.data))

def train(enc, gen, dis, optimizer_enc, optimizer_gen, optimizer_dis, epoch_num, gpu_device=None, out_image_dir=None):
    if gpu_device == None:
        enc.to_cpu()
        gen.to_cpu()
        dis.to_cpu()
        xp = np
    else:
        enc.to_gpu(gpu_device)
        gen.to_gpu(gpu_device)
        dis.to_gpu(gpu_device)
        xp = cuda.cupy
    out_image_row_num = 10
    out_image_col_num = 10
    z_out_image =  Variable(xp.random.uniform(-1, 1, (out_image_row_num * out_image_col_num, latent_size)).astype(np.float32))
    x_batch = np.zeros((BATCH_SIZE, 3, image_size, image_size), dtype=np.float32)
    for epoch in xrange(1, epoch_num + 1):
        x_size = len(images)
        perm = np.random.permutation(x_size)
        sum_loss_enc = 0
        sum_loss_gen = 0
        sum_loss_dis = 0
        for i in xrange(0, x_size, BATCH_SIZE):
            x_batch.fill(0)
            for j, p in enumerate(perm[i:i + BATCH_SIZE]):
                image = images[p]
                offset_x = np.random.randint(8) + 37
                offset_y = np.random.randint(8) + 68
                w = 96
                h = 96
                with io.BytesIO(image) as b:
                    pixels = np.asarray(Image.open(b).convert('RGB').crop((offset_x, offset_y, offset_x + w, offset_y + h)).resize((image_size, image_size)))
                    pixels = pixels.astype(np.float32).transpose((2, 0, 1)).reshape((3, image_size, image_size))
                    x_batch[j] = pixels / 127.5 - 1
            loss_enc, loss_gen, loss_dis = train_one(enc, gen, dis, optimizer_enc, optimizer_gen, optimizer_dis, x_batch, gpu_device)
            sum_loss_enc += loss_enc * BATCH_SIZE
            sum_loss_gen += loss_gen * BATCH_SIZE
            sum_loss_dis += loss_dis * BATCH_SIZE
            if i % image_save_interval == 0:
                print '{} {} {}'.format(sum_loss_enc / (i + BATCH_SIZE), sum_loss_gen / (i + BATCH_SIZE), sum_loss_dis / (i + BATCH_SIZE))
                if out_image_dir != None:
                    z, m, v = enc(Variable(xp.asarray(x_batch), volatile=True), train=False)
                    data = gen(z, train=False).data
#                    data = gen((z_out_image), train=False).data
                    image = ((cuda.to_cpu(data) + 1) * 128).clip(0, 255).astype(np.uint8)
                    image = image.reshape((out_image_row_num, out_image_col_num, 3, image_size, image_size)).transpose((0, 3, 1, 4, 2)).reshape((out_image_row_num * image_size, out_image_col_num * image_size, 3))
                    Image.fromarray(image).save('{0}/{1:03d}_{2:07d}.png'.format(out_image_dir, epoch, i))
                    org_image = ((x_batch + 1) * 128).clip(0, 255).astype(np.uint8)
                    org_image = org_image.reshape((out_image_row_num, out_image_col_num, 3, image_size, image_size)).transpose((0, 3, 1, 4, 2)).reshape((out_image_row_num * image_size, out_image_col_num * image_size, 3))
                    Image.fromarray(org_image).save('{0}/{1:03d}_{2:07d}_org.png'.format(out_image_dir, epoch, i))
        print 'epoch: {} done'.format(epoch)
        print('enc loss={}'.format(sum_loss_enc / x_size))
        print('gen loss={}'.format(sum_loss_gen / x_size))
        print('dis loss={}'.format(sum_loss_dis / x_size))
        serializers.save_hdf5('{0}_{1:03d}.enc.model'.format(args.output, epoch), enc)
        serializers.save_hdf5('{0}_{1:03d}.enc.state'.format(args.output, epoch), optimizer_enc)
        serializers.save_hdf5('{0}_{1:03d}.gen.model'.format(args.output, epoch), gen)
        serializers.save_hdf5('{0}_{1:03d}.gen.state'.format(args.output, epoch), optimizer_gen)
        serializers.save_hdf5('{0}_{1:03d}.dis.model'.format(args.output, epoch), dis)
        serializers.save_hdf5('{0}_{1:03d}.dis.state'.format(args.output, epoch), optimizer_dis)

train(enc_model, gen_model, dis_model, optimizer_enc, optimizer_gen, optimizer_dis, args.iter, gpu_device=gpu_device, out_image_dir=args.out_image_dir)
