import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple
import tensorflow_probability as tfp
tfd = tfp.distributions

os.environ['CUDA_VISIBLE_DEVICES']= '-1'


'''
summary of leaning task:
    Learn a (possibly recurrent) neural network that computes the shaping
    parameters of the variational posterior at the current time step given... 
     - the shaping parameters at the previous time step, and
     - the current observations y(t) and x(t)

observation model:
    p(y(t) | z(t), x(t)) = Normal(z(t)*x(t), R)

dynamic model:
    p(z(t) | z(t-1)) = Normal(z(t-1), Q)

joint:
    p(y(t), z(t), z(t-1) | x(t))
        = p(y(t) | z(t), x(t)) * p(z(t) | z(t-1)) * q(z(t-1) | y(t-1), x(t-1))

variational posterior:
    q(z(t), z(t-1) | y(t), x(t))

note:
    the factored joint distribution above uses the marginalized variational 
    posterior from the previous time step...
         q(z(t-1) | y(t-1), x(t-1))


'''


EncoderInputs = namedtuple(
    'EncoderInputs',
    [
        'y',
        'x',
    ]
)

EncoderStates = namedtuple(
    'EncoderStates',
    [
        'h_mu', # mean 
        'h_L'   # elements of lower-tri. scale matrix
    ]
)

EncoderParams = namedtuple(
    'EncoderParams',
    [
        'W1',
        'b1',
        'W2'
    ]
)

def encoder(inputs, initial_states, params):
    
    assert isinstance(inputs, EncoderInputs)
    assert isinstance(initial_states, EncoderStates)
    assert isinstance(params, EncoderParams)

    def encoder_inner(states, ins):

        yx = tf.stack([ins.y, ins.x], axis=-1)

        u = tf.concat(
            [states.h_mu, states.h_L, yx],
            axis=-1
        )

        a = tf.nn.relu(tf.tensordot(params.W1, u, axes=1) + params.b1)
        h = tf.tensordot(params.W2, a, axes=1)

        next_states = EncoderStates(h_mu=h[0:2], h_L=h[2:])

        return next_states

    result = tf.scan(
        encoder_inner,
        inputs,
        initializer=initial_states
    )

    return result


if __name__ == '__main__':

    ''' 
    make fake data...

    phi(t) = 2*pi*t/42
    x(t) = sin(phi(t))
    z(t) = z(t-1) + w(t), w(t) ~ Normal(0, Q)
    y(t) = z(t)*x(t) + v(t), v(t) ~ Normal(0, R)

    '''
    Q = 0.1
    R = 0.1
    z0 = 1.0

    Npts = 500
    Ncycles = 3
    phi = 2*np.pi*tf.range(Npts, dtype=tf.float32)/(Npts/Ncycles)
    x = tf.math.sin(phi)
    z = tf.scan(
        lambda acc, a: acc + a,
        tf.random.normal([Npts], mean=0, stddev=tf.math.sqrt(Q)),
        initializer=z0
    )
    y = z*x + tf.random.normal([Npts], mean=0, stddev=tf.math.sqrt(R))

    # encoder to estimate shaping parameters of variational posterior
    triL_mask = tfp.math.fill_triangular(tf.ones([3], dtype=tf.bool))
    h_mu = tf.ones([2])
    h_L = tf.boolean_mask(tf.eye(2), triL_mask)

    num_in = 7
    num_hidden = 20
    num_out = 5

    W1 = tf.Variable(tf.zeros([num_hidden, num_in]))
    b1 = tf.Variable(tf.zeros([num_hidden]))
    W2 = tf.Variable(tf.zeros([num_out, num_hidden]))

    inputs = EncoderInputs(y=y, x=x)
    initial_states = EncoderStates(h_mu=h_mu, h_L=h_L)
    params = EncoderParams(W1=W1, b1=b1, W2=W2)

    outputs = encoder(inputs, initial_states, params)

    print('Done!')