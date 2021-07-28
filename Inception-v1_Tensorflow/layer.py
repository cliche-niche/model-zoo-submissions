import tensorflow as tf
from tensorflow.keras import layers


class reduce(layers.Layer):
    def __init__(self, filter1x1, ker_size, filters):
        super(reduce, self).__init__()
        self.con1 = layers.Conv2D(filter1x1, kernel_size=1, padding="same", activation="relu")
        self.conv = layers.Conv2D(filters, kernel_size=ker_size, padding="same", activation="relu")

    def call(self, inp):
        x = self.con1(inp)
        x = self.conv(x)
        return x



class poolproj(layers.Layer):
    def __init__(self, filter1x1):
        super(poolproj, self).__init__()
        self.max = layers.MaxPooling2D(pool_size=3, strides=1, padding="same")
        self.conv = layers.Conv2D(filter1x1, kernel_size=1, padding="same", activation="relu")

    def call(self, inp):
        x = self.max(inp)
        x = self.conv(x)
        return x



class inceptionblock(layers.Layer):
    def __init__(self, filter1x1, red3, red5, pool):
        super(inceptionblock, self).__init__()
        self.conv1 = layers.Conv2D(filter1x1, kernel_size=1, padding="same", activation="relu")
        self.conv3 = reduce(red3[0], red3[1], red3[2])
        self.conv5 = reduce(red5[0], red5[1], red5[2])
        self.poolp = poolproj(pool)

    def call(self, inp):
        x1 = self.conv1(inp)
        x2 = self.conv3(inp)
        x3 = self.conv5(inp)
        x4 = self.poolp(inp)
        return tf.concat([x1, x2, x3, x4], 3)


class auxiliary(layers.Layer):
    def __init__(self, channels=10):
        super(auxiliary, self).__init__()
        self.avg = layers.AveragePooling2D(pool_size=5, strides=3)
        self.con1 = layers.Conv2D(128, kernel_size=1, padding="same", activation="relu")
        self.flat = layers.Flatten()
        self.full1 = layers.Dense(1024, activation="relu")
        self.drop = layers.Dropout(0.7)
        self.full2 = layers.Dense(channels, activation="softmax")

    def call(self, inp):
        x = self.avg(inp)
        x = self.con1(x)
        x = self.flat(x)
        x = self.full1(x)
        x = self.drop(x)
        x = self.full2(x)
        return x