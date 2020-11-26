"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import numpy as np
from tensorflow.keras import Sequential

from ops import *


class Generator(tf.keras.Model):
    def __init__(self, img_size=256, img_ch=3, style_dim=64, max_conv_dim=512, sn=False, name='Generator'):
        super(Generator, self).__init__(name=name)
        self.img_size = img_size
        self.img_ch = img_ch
        self.style_dim = style_dim
        self.max_conv_dim = max_conv_dim
        self.sn = sn

        self.channels = 2 ** 14 // img_size  # if 256 -> 64
        self.repeat_num = int(np.log2(img_size)) - 4  # if 256 -> 4

        self.from_rgb = Conv(channels=self.channels, kernel=3, stride=1, pad=1, sn=self.sn, name='from_rgb')
        self.to_rgb = Sequential(
            [
                InstanceNorm(),
                Leaky_Relu(alpha=0.2),
                Conv(channels=self.img_ch, kernel=1, stride=1, sn=self.sn, name='to_rgb')
            ]
        )

        self.encoder, self.decoder = self.architecture_init()

    def architecture_init(self):
        ch_in = self.channels
        ch_out = self.channels

        encoder = []
        decoder = []

        # down/up-sampling blocks
        self.adains = []
        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim)

            encoder.append(ResBlock(ch_in, ch_out, normalize=True, downsample=True, sn=self.sn,
                                    name='encoder_down_resblock_' + str(i)))
            decoder.insert(0, DecoderResBlock(ch_out, ch_in, upsample=True, sn=self.sn, adain=False,
                                              name='decoder_up_resblock_' + str(i)))  # stack-like
            self.adains.insert(0, False)

            ch_in = ch_out

        # bottleneck blocks
        assert 6 - self.repeat_num == 2
        for i in range(self.repeat_num, 6):
            encoder.append(
                ResBlock(ch_out, ch_out, normalize=True, downsample=True, sn=self.sn,
                         name='encoder_down_resblock_' + str(i)))
            decoder.insert(0,
                           DecoderResBlock(ch_out, ch_out, upsample=True, sn=self.sn, adain=True,
                                           name='decoder_up_adaresblock_' + str(i)))
            self.adains.insert(0, True)
        return encoder, decoder

    def call(self, x_init, training=True, mask=None):
        x, x_s, x_mask = x_init
        rgb_skip = x
        x_mask = tf.cast(x_mask, dtype=x.dtype)
        x_mask = x_mask[..., None]
        x = tf.concat([x, x_mask], axis=-1)

        x = self.from_rgb(x)

        for encoder_block in self.encoder:
            x = encoder_block(x)

        for decoder_block, adain in zip(self.decoder, self.adains):
            if adain:
                decoder_input = [x, x_s]
            else:
                decoder_input = [x, None]
            x = decoder_block(decoder_input)

        x = self.to_rgb(x) * x_mask + rgb_skip

        return x


class MappingNetwork(tf.keras.Model):
    def __init__(self, style_dim=64, hidden_dim=512, sn=False, name='MappingNetwork'):
        super(MappingNetwork, self).__init__(name=name)
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.sn = sn

        self.mapping_network_layers = self.architecture_init()

    def architecture_init(self):
        layers = []
        layers += [FullyConnected(units=self.hidden_dim, sn=self.sn, name='shared_fc')]
        layers += [Relu()]

        for i in range(6):
            layers += [FullyConnected(units=self.hidden_dim, sn=self.sn, name='shared_fc_' + str(i))]
            layers += [Relu()]
        layers += [FullyConnected(units=self.style_dim, sn=self.sn, name='style_fc')]

        return Sequential(layers)

    def call(self, inputs, training=True, mask=None):
        z = inputs
        return self.mapping_network_layers(z)


class StyleEncoder(tf.keras.Model):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, sn=False, name='StyleEncoder'):
        super(StyleEncoder, self).__init__(name=name)
        self.img_size = img_size
        self.style_dim = style_dim
        self.max_conv_dim = max_conv_dim
        self.sn = sn

        self.channels = 2 ** 14 // img_size  # if 256 -> 64
        self.repeat_num = int(np.log2(img_size)) - 2  # if 256 -> 6

        self.style_encoder_layers = self.architecture_init()

    def architecture_init(self):
        # shared layers
        ch_in = self.channels
        ch_out = self.channels
        blocks = []

        blocks += [Conv(ch_in, kernel=3, stride=1, pad=1, sn=self.sn, name='init_conv')]

        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim)

            blocks += [ResBlock(ch_in, ch_out, downsample=True, sn=self.sn, name='resblock_' + str(i))]
            ch_in = ch_out

        blocks += [Leaky_Relu(alpha=0.2)]
        blocks += [Conv(channels=ch_out, kernel=4, stride=1, pad=0, sn=self.sn, name='conv')]
        blocks += [Leaky_Relu(alpha=0.2)]
        blocks += [FullyConnected(units=self.style_dim, sn=self.sn, name='style_fc')]

        return Sequential(blocks)

    def call(self, inputs, training=True, mask=None):
        x = inputs
        return self.style_encoder_layers(x)


class Discriminator(tf.keras.Model):
    def __init__(self, img_size=256, max_conv_dim=512, sn=False, name='Discriminator'):
        super(Discriminator, self).__init__(name=name)

        self.img_size = img_size
        self.max_conv_dim = max_conv_dim
        self.sn = sn

        self.channels = 2 ** 14 // img_size  # if 256 -> 64
        self.repeat_num = int(np.log2(img_size)) - 2  # if 256 -> 6

        self.encoder = self.architecture_init()

    def architecture_init(self):
        ch_in = self.channels
        ch_out = self.channels
        blocks = []

        blocks += [Conv(ch_in, kernel=3, stride=1, pad=1, sn=self.sn, name='init_conv')]

        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim // 2 if i < 5 else self.max_conv_dim)

            blocks += [ResBlock(ch_in, ch_out, downsample=True, sn=self.sn, name='resblock_' + str(i))]

            ch_in = ch_out

        blocks += [Leaky_Relu(alpha=0.2)]
        blocks += [Conv(channels=ch_out, kernel=4, stride=1, pad=0, sn=self.sn, name='conv_0')]

        blocks += [Leaky_Relu(alpha=0.2)]
        blocks += [Conv(channels=1, kernel=1, stride=1, sn=self.sn, name='conv_1')]

        encoder = Sequential(blocks)

        return encoder

    def call(self, inputs, training=True, mask=None):
        x, x_mask = inputs
        x_mask = tf.cast(x_mask, dtype=x.dtype)
        x_mask = x_mask[..., None]
        x = tf.concat([x, x_mask], axis=-1)

        x = self.encoder(x)
        x = tf.reshape(x, shape=[x.shape[0], -1])  # [bs, 1]

        return x
