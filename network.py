import numpy as np
import sonnet as snt
import tensorflow as tf

from sonnet.python.modules import base
from tensorflow.python.training import moving_averages

from dataset import Dataset


class Encoder(snt.AbstractModule):
    def __init__(self, latent_space_size, num_filters, kernel_size, strides, batch_norm, gen_obj_code, name='encoder'):
        super(Encoder, self).__init__(name=name)
        self._latent_space_size = latent_space_size
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._batch_normalization = batch_norm
        self._gen_obj_code=gen_obj_code

    @property
    def latent_space_size(self):
        return self._latent_space_size

    @property
    def encoder_layers(self):
        layers = []
        x = self._input
        layers.append(x)

        for filters, stride in zip(self._num_filters, self._strides):
            padding = 'same'
            x = tf.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=self._kernel_size,
                strides=stride,
                padding=padding,
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=tf.nn.relu,
                # name='ecd_{:d}'.format(_id)
            )
            if self._batch_normalization:
                x = tf.layers.batch_normalization(x, training=self._is_training)
            layers.append(x)
        return layers

    @property
    def encoder_out(self):
        x = self.encoder_layers[-1]
        encoder_out = tf.contrib.layers.flatten(x)
        return encoder_out

    @property
    def z(self):
        x = self.encoder_out
        z = tf.layers.dense(
            x,
            self._latent_space_size,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name=None
        )
        return z

    def _build(self, x, is_training=False):
        self._input = x
        self._is_training = is_training
        return self.z

class Decoder(snt.AbstractModule):
    def __init__(self, reconstruction_shape, num_filters,kernel_size, strides,
                 auxiliary_mask, batch_norm, name='decoder'):
        super(Decoder, self).__init__(name=name)
        self._reconstruction_shape=reconstruction_shape
        self._auxiliary_mask = auxiliary_mask
        if self._auxiliary_mask:
            self._xmask = None
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._batch_normalization = batch_norm

    def _build(self, latent_code,is_training=False):
        z=latent_code
        h, w, c = self._reconstruction_shape[0:3]
        print(h,w,c)
        layer_dimensions = [ [h//np.prod(self._strides[i:]), w//np.prod(self._strides[i:])]  for i in range(0,len(self._strides))]
        print(layer_dimensions)

        if c!=1:
            with tf.variable_scope('decoder_bgr'):
                x = tf.layers.dense(
                    inputs=z,
                    units= layer_dimensions[0][0]*layer_dimensions[0][1]*self._num_filters[0],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer()
                )
                if self._batch_normalization:
                    x = tf.layers.batch_normalization(x, training=self._is_training)
                x = tf.reshape( x, [-1, layer_dimensions[0][0], layer_dimensions[0][1], self._num_filters[0] ] )

                for filters, layer_size in zip(self._num_filters[1:],layer_dimensions[1:]):
                    x = tf.image.resize_nearest_neighbor(x, layer_size)
                    x = tf.layers.conv2d(
                        inputs=x,
                        filters=filters,
                        kernel_size=self._kernel_size,
                        padding='same',
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        activation=tf.nn.relu
                    )
                    if self._batch_normalization:
                        x = tf.layers.batch_normalization(x, training=self._is_training)

                x = tf.image.resize_nearest_neighbor( x, [h, w] )

                if self._auxiliary_mask:
                    self._xmask = tf.layers.conv2d(
                            inputs=x,
                            filters=1,
                            kernel_size=self._kernel_size,
                            padding='same',
                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            activation=tf.nn.sigmoid
                        )

                x_bgr = tf.layers.conv2d(
                        inputs=x,
                        filters=3,
                        kernel_size=self._kernel_size,
                        padding='same',
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        activation=tf.nn.sigmoid
                    )
        else:
            x_bgr=None

        if c!=3:
            with tf.variable_scope('decoder_edge'):
                x = tf.layers.dense(
                    inputs=z,
                    units=layer_dimensions[0][0] * layer_dimensions[0][1] * self._num_filters[0],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer()
                )
                if self._batch_normalization:
                    x = tf.layers.batch_normalization(x, training=self._is_training)
                x = tf.reshape(x, [-1, layer_dimensions[0][0], layer_dimensions[0][1], self._num_filters[0]])

                _id = len(layer_dimensions[1:])
                for filters, layer_size in zip(self._num_filters[1:], layer_dimensions[1:]):
                    x = tf.image.resize_nearest_neighbor(x, layer_size)
                    x = tf.layers.conv2d(
                        inputs=x,
                        filters=filters,
                        kernel_size=self._kernel_size,
                        padding='same',
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        activation=tf.nn.relu,
                    )
                    _id -= 1
                    if self._batch_normalization:
                        x = tf.layers.batch_normalization(x, training=self._is_training)

                x = tf.image.resize_nearest_neighbor(x, [h, w])

                if self._auxiliary_mask:
                    self._xmask = tf.layers.conv2d(
                        inputs=x,
                        filters=1,
                        kernel_size=self._kernel_size,
                        padding='same',
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        activation=tf.nn.sigmoid
                    )

                activation_final = None
                x_edge = tf.layers.conv2d(
                    inputs=x,
                    filters=1,
                    kernel_size=self._kernel_size,
                    padding='same',
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    activation=activation_final
                )
                x_edge_vis=tf.nn.sigmoid(x_edge)
        else:
            x_edge=None
            x_edge_vis=None

        return {'x_bgr':x_bgr,
                'x_edge':x_edge,
                'x_edge_vis':x_edge_vis}


def build_encoder(args,gen_obj_code=False):
    LATENT_SPACE_SIZE = args.getint('Network', 'LATENT_SPACE_SIZE')
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_ENCODER = args.getint('Network', 'KERNEL_SIZE_ENCODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')
    encoder = Encoder(
        latent_space_size=LATENT_SPACE_SIZE,
        num_filters=NUM_FILTER,
        kernel_size=KERNEL_SIZE_ENCODER,
        strides=STRIDES,
        batch_norm=BATCH_NORM,
        gen_obj_code=gen_obj_code
    )
    return encoder


def build_decoder(args):
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_DECODER = args.getint('Network', 'KERNEL_SIZE_DECODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    AUXILIARY_MASK = args.getboolean('Network', 'AUXILIARY_MASK')
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')

    R_H = args.getint('Dataset', 'H')
    R_W = args.getint('Dataset', 'W')
    R_CO = args.getint('Dataset', 'CO')

    decoder = Decoder(
        reconstruction_shape=[R_H, R_W, R_CO],
        num_filters=list(reversed(NUM_FILTER)),
        kernel_size=KERNEL_SIZE_DECODER,
        strides=list(reversed(STRIDES)),
        auxiliary_mask=AUXILIARY_MASK,
        batch_norm=BATCH_NORM,
    )

    return decoder

def build_dataset(dataset_path,fg_path_format,codebook_path_format,list_objs, args):
    dataset_args = { k:v for k,v in
        args.items('Dataset') +
        args.items('Augmentation')}
    dataset = Dataset(dataset_path,fg_path_format,codebook_path_format,list_objs, **dataset_args)
    return dataset

class VectorQuantizerEMA(base.AbstractModule):
    """
    decay: float, decay for the moving averages.
    epsilon: small float constant to avoid numerical instability.
    w: is a matrix with an embedding in each column. When training, the embedding is assigned to be the average of all inputs assigned to that embedding.
    """
    def __init__(self, embedding_dim, num_embeddings, epsilon=1e-5,name='vq_center'):
        super(VectorQuantizerEMA, self).__init__(name=name)
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._epsilon = epsilon

        with self._enter_variable_scope():
          initializer = tf.random_normal_initializer()
          self._w = tf.get_variable('embedding', [embedding_dim, num_embeddings],initializer=initializer, use_resource=True)
          self._ema_cluster_size = tf.get_variable('ema_cluster_size', [num_embeddings],initializer=tf.constant_initializer(1), use_resource=True)#0
          self._ema_w = tf.get_variable('ema_dw', initializer=self._w.initialized_value(), use_resource=True)

    def _build(self, inputs, decay=0.99, temperature=0.07, encoding_1nn_indices=None, encodings=None, mask_roi=None, is_training=False):
        with tf.control_dependencies([inputs]):
            w = self._w.read_value()

        #mask_roi is a tuple, that consider only a certain range of codes
        if not (mask_roi is None):
            masked_w = w[:,mask_roi[0]:mask_roi[1]]
        else:
            masked_w = w

        input_shape = tf.shape(inputs)

        #Flat inputs
        with tf.control_dependencies([tf.Assert(tf.equal(input_shape[-1], self._embedding_dim),[input_shape])]):
            flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])

        #Test stage: compute the index with minimal cosine error
        if encoding_1nn_indices is None:
            distances = -tf.matmul(tf.nn.l2_normalize(flat_inputs, axis=1), tf.nn.l2_normalize(masked_w, axis=0))
            encoding_1nn_indices = tf.argmax(-distances, 1)
            if not (mask_roi is None):
                encoding_1nn_indices+=mask_roi[0]

        if encodings is None:
            encodings = tf.squeeze(tf.one_hot(encoding_1nn_indices, self._num_embeddings))
            print('VQEMA: Encodings 1NN shape',encoding_1nn_indices.shape, 'Encodings is ONE-HOT with shape',encodings.shape)
        #Reshape Encoding 1nn indices for loss computation
        encoding_1nn_indices = tf.reshape(encoding_1nn_indices, tf.shape(inputs)[:-1])

        quantized_1nn=self.quantize(encoding_1nn_indices)

        normalized_inputs=tf.nn.l2_normalize(flat_inputs,axis=1)
        normalized_w=tf.nn.l2_normalize(w,axis=0)
        e_multiply=tf.matmul(normalized_inputs,tf.stop_gradient(normalized_w))/temperature

        flat_encodings=tf.reshape(encodings,[-1,self._num_embeddings])
        print('Flatten Encodings with Shape',flat_encodings.shape,'// Prediction shape',e_multiply.shape)
        e_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(flat_encodings),logits=e_multiply))

        if is_training:#w,dw [128,8020], input batch [64,128]
            updated_ema_cluster_size = moving_averages.assign_moving_average(self._ema_cluster_size, tf.reduce_sum(encodings, 0), decay,zero_debias=False)
            print('Dw shape',normalized_inputs.shape,'Encoding shape',encodings.shape)
            dw = tf.matmul(normalized_inputs, encodings, transpose_a=True)
            updated_ema_w = moving_averages.assign_moving_average(self._ema_w, dw, decay,zero_debias=False)
            n = tf.reduce_sum(updated_ema_cluster_size)

            normalised_updated_ema_w = (updated_ema_w / tf.reshape(updated_ema_cluster_size, [1, -1]))
            with tf.control_dependencies([e_loss]):
                update_w = tf.assign(self._w, normalised_updated_ema_w)
                with tf.control_dependencies([update_w]):
                    loss = tf.identity(e_loss)
        else:
            loss = e_loss
        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))

        return {'quantize_1nn': quantized_1nn,
                'loss': loss,
                'perplexity': perplexity,
                'encodings': encodings,
                'encoding_indices': encoding_1nn_indices, }

    @property
    def embeddings(self):
        return self._w

    def quantize(self, encoding_indices):
        with tf.control_dependencies([encoding_indices]):
            w = tf.transpose(self.embeddings.read_value(), [1, 0])
        return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)
