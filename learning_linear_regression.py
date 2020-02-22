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
     - the current observations x and y

notation:
    - x, y, z is shorthand for x(t), y(t), z(t)
    - X is shorthand for the set {x(1), x(2), ..., x(t)}, similarly for Y
    - x', y', z', is shorthand for x(t-1), y(t-1), z(t-1)
    - X' is shorthand for the set {x(1), x(2), ..., x(t-1)}, similarly for Y'

observation model:
    p(y | z, x) = Normal(z*x, R)

dynamic model:
    p(z | z') = Normal(z', Q)

joint:
    p(y, z, z' | Y', X)
        = p(y | z, x) * p(z | z') * q(z' | Y', X')

variational posterior:
    q(z, z' | Y, X)

note:
    the factored joint distribution above uses the marginalized variational 
    posterior from the previous time step...
         q(z' | Y', X')


'''


EncoderInputs = namedtuple(
    'EncoderInputs',
    [
        'Y',
        'X',
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

def encoder(inputs, initial_state, params):
    
    assert isinstance(inputs, EncoderInputs)
    assert isinstance(initial_states, EncoderStates)
    assert isinstance(params, EncoderParams)

    def encoder_inner(encoder_state, encoder_input):
        # assume encoder_state has type EncoderStates
        # assume encoder_input has type EncoderInputs

        yx = tf.stack([encoder_input.Y, encoder_input.X], axis=-1)

        u = tf.concat(
            [encoder_state.h_mu, encoder_state.h_L, yx],
            axis=-1
        )

        a = tf.nn.tanh(tf.tensordot(params.W1, u, axes=1) + params.b1)
        h = tf.tensordot(params.W2, a, axes=1)

        encoder_output = EncoderStates(h_mu=h[0:2], h_L=h[2:])

        return encoder_output

    result = tf.scan(
        encoder_inner,
        inputs,
        initializer=initial_state
    )

    # construct variational posterior from shaping parameters in result
    qz = tfd.Independent(
        distribution=tfd.MultivariateNormalTriL(
            loc=result.h_mu,
            scale_tril=tfp.math.fill_triangular(result.h_L)
        ),
        reinterpreted_batch_ndims=1
    )

    return qz


DecoderParams = namedtuple(
    'DecoderParams',
    [
        'R',
        'Q',
        'L0',
        'mu0'
    ]
)

def decoder(codes, inputs, params, qz):
    # codes are samples z, z' ~ q(z, z'|Y, X)

    assert(isinstance(codes, tf.Tensor))
    assert(isinstance(inputs, EncoderInputs))
    assert(isinstance(params, DecoderParams))
    assert(isinstance(qz, tfp.distributions.Distribution))

    def observation_density(codes, inputs, params):
        # p(y(t) | z(t), x(t)) = Normal(z(t)*x(t), R)
        py = tfd.Normal(loc=codes[:,0]*inputs.X, scale=tf.math.sqrt(params.R))

        return py

    def transition_density(codes, params):
        # p(z(t) | z(t-1)) = Normal(z(t-1), Q)
        pz = tfd.Normal(loc=codes[:,1], scale=tf.math.sqrt(params.Q))

        return pz

    def prior_density(params, qz):
        # q(z(t-1) | y(t-1), x(t-1))
        # the marginalized variational posterior from the previous time step
        loc = tf.concat(
            [tf.expand_dims(params.mu0[0], axis=0), qz.distribution.loc[0:-1,0]],
            axis=0
        )

        scale = tf.concat(
            [tf.expand_dims(params.L0[0,0], axis=0), qz.distribution.scale_tril[0:-1,0,0]],
            axis=0
        )

        qzm = tfd.Normal(loc=loc, scale=scale) 

        return qzm

    py = observation_density(codes, inputs, params)
    pz = transition_density(codes, params)
    qzm = prior_density(params, qz)
        

    return py, pz, qzm


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
    mu0 = tf.zeros([2])
    L0 = tf.eye(2)
    triL_mask = tfp.math.fill_triangular(tf.ones([3], dtype=tf.bool))

    num_in = 7 # x, y, h_mu (2), h_L (3)
    num_hidden = 20
    num_out = 5 # h_mu (2), h_L (3)

    W1 = tf.Variable(tf.random.normal([num_hidden, num_in]))
    b1 = tf.Variable(tf.zeros([num_hidden]))
    W2 = tf.Variable(tf.random.normal([num_out, num_hidden]))

    inputs = EncoderInputs(Y=y, X=x)
    initial_states = EncoderStates(h_mu=mu0, h_L=tf.boolean_mask(L0, triL_mask))
    params = EncoderParams(W1=W1, b1=b1, W2=W2)
    dparams = DecoderParams(R=R, Q=Q, L0=L0, mu0=mu0)

    qz = encoder(inputs, initial_states, params)
    codes = qz.sample()
    py, pz, qzm = decoder(codes, inputs, dparams, qz)


    print('Done!')