"""
CBAM (Convolutional Block Attention Module) layer definitions.

These MUST be byte-for-byte identical to the classes used when the model
was trained (see the model development notebook's "CBAM Attention Module"
section) -- Keras needs the exact same custom layer code to deserialize
the saved .keras file. If you change anything here, you'll need to
re-save the trained model or loading will fail / silently misbehave.
"""
import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(1, channels // self.reduction_ratio)
        self.dense1 = layers.Dense(reduced, activation='relu', use_bias=False)
        self.dense2 = layers.Dense(channels, activation=None, use_bias=False)
        super().build(input_shape)

    def call(self, x):
        avg = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        avg = self.dense2(self.dense1(avg))
        mx = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        mx = self.dense2(self.dense1(mx))
        scale = tf.sigmoid(avg + mx)
        return x * scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'reduction_ratio': self.reduction_ratio})
        return cfg


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(
            filters=1, kernel_size=kernel_size,
            padding='same', activation='sigmoid', use_bias=False
        )

    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx = tf.reduce_max(x, axis=-1, keepdims=True)
        combined = tf.concat([avg, mx], axis=-1)
        return x * self.conv(combined)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'kernel_size': self.kernel_size})
        return cfg


class CBAM(layers.Layer):
    def __init__(self, reduction_ratio=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.channel_att = ChannelAttention(reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def call(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'reduction_ratio': self.reduction_ratio, 'kernel_size': self.kernel_size})
        return cfg


class BinaryFocalLoss(tf.keras.losses.Loss):
    """Only needed here because the saved model was compiled with it at
    some point in training; compile=False at load time means this is
    mostly for completeness/safety, not actually used at inference."""

    def __init__(self, alpha=0.35, gamma=1.0, name='binary_focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_w = alpha_t * tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(focal_w * bce)

    def get_config(self):
        return {'alpha': self.alpha, 'gamma': self.gamma, 'name': self.name}


CUSTOM_OBJECTS = {
    'BinaryFocalLoss': BinaryFocalLoss,
    'CBAM': CBAM,
    'ChannelAttention': ChannelAttention,
    'SpatialAttention': SpatialAttention,
}
