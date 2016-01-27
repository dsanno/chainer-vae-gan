import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

class Encoder(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        assert(size % 16 == 0)
        initial_size = size / 16
        super(Encoder, self).__init__(
            dc1   = L.Convolution2D(channel, 32 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * channel * density)),
            dc2   = L.Convolution2D(32 * density, 64 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
            norm2 = L.BatchNormalization(64 * density),
            dc3   = L.Convolution2D(64 * density, 128 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm3 = L.BatchNormalization(128 * density),
            dc4   = L.Convolution2D(128 * density, 256 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128 * density)),
            norm4 = L.BatchNormalization(256 * density),
            mean  = L.Linear(initial_size * initial_size * 256 * density, latent_size, wscale=0.02 * math.sqrt(initial_size * initial_size * 256 * density)),
            var   = L.Linear(initial_size * initial_size * 256 * density, latent_size, wscale=0.02 * math.sqrt(initial_size * initial_size * 256 * density)),
        )

    def __call__(self, x, train=True):
        xp = cuda.get_array_module(x.data)
        h1 = F.leaky_relu(self.dc1(x))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        mean = self.mean(h4)
        var  = self.var(h4)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + F.exp(var) * Variable(rand, volatile=not train)
        return (z, mean, var)

class Generator(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        assert(size % 16 == 0)
        initial_size = size / 16
        super(Generator, self).__init__(
            g1    = L.Linear(latent_size, initial_size * initial_size * 256 * density, wscale=0.02 * math.sqrt(latent_size)),
            norm1 = L.BatchNormalization(initial_size * initial_size * 256 * density),
            g2    = L.Deconvolution2D(256 * density, 128 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256 * density)),
            norm2 = L.BatchNormalization(128 * density),
            g3    = L.Deconvolution2D(128 * density, 64 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128 * density)),
            norm3 = L.BatchNormalization(64 * density),
            g4    = L.Deconvolution2D(64 * density, 32 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm4 = L.BatchNormalization(32 * density),
            g5    = L.Deconvolution2D(32 * density, channel, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
        )
        self.density = density
        self.latent_size = latent_size
        self.initial_size = initial_size

    def __call__(self, z, train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)), (z.data.shape[0], 256 * self.density, self.initial_size, self.initial_size))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
        return F.tanh(self.g5(h4))

class Discriminator(chainer.Chain):
    def __init__(self, density=1, size=64, channel=3):
        assert(size % 16 == 0)
        initial_size = size / 16
        super(Discriminator, self).__init__(
            dc1   = L.Convolution2D(channel, 32 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * channel * density)),
            dc2   = L.Convolution2D(32 * density, 64 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
            norm2 = L.BatchNormalization(64 * density),
            dc3   = L.Convolution2D(64 * density, 128 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm3 = L.BatchNormalization(128 * density),
            dc4   = L.Convolution2D(128 * density, 256 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128 * density)),
            norm4 = L.BatchNormalization(256 * density),
            dc5   = L.Linear(initial_size * initial_size * 256 * density, 2, wscale=0.02 * math.sqrt(initial_size * initial_size * 256 * density)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.dc1(x))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        return (self.dc5(h4), h3)

class Generator48(chainer.Chain):
    def __init__(self):
        latent_size = 100
        super(Generator48, self).__init__(
            g1    = L.Linear(latent_size * 2, 6 * 6 * 256, wscale=0.02 * math.sqrt(latent_size)),
            norm1 = L.BatchNormalization(6 * 6 * 256),
            g2    = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            norm2 = L.BatchNormalization(128),
            g3    = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm3 = L.BatchNormalization(64),
            g4    = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
        )
        self.latent_size = latent_size

    def __call__(self, (z, y), train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)), (z.data.shape[0], 256, 6, 6))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.sigmoid(self.g4(h3))
        return h4

class Discriminator48(chainer.Chain):
    def __init__(self):
        super(Discriminator48, self).__init__(
            dc1   = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(64),
            dc2   = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm2 = L.BatchNormalization(128),
            dc3   = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm3 = L.BatchNormalization(256),
            dc4   = L.Linear(6 * 6 * 256, 2, wscale=0.02 * math.sqrt(6 * 6 * 256)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        return self.dc4(h3)
