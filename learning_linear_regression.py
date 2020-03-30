import os, time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple
import tensorflow_probability as tfp
tfd = tfp.distributions

os.environ['CUDA_VISIBLE_DEVICES']= '0'

tf.random.set_seed(5678)



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

def make_encoder(batch_size, sequence_len, hidden_dim=16):

    unroll = False
    state_dim = 5
    layers = []

    # add input lstm layer
    layers.append(
        tf.keras.layers.LSTM(
            units=state_dim,
            return_sequences=True,
            stateful=True,
            batch_input_shape=[batch_size, sequence_len, 2],
            unroll=unroll
        )
    )

    # add second lstm layer
    layers.append(
        tf.keras.layers.LSTM(
            units=hidden_dim,
            return_sequences=True,
            unroll=unroll
        )
    )

    # # add first output layer
    # layers.append(
    #     tf.keras.layers.Dense(
    #         units=hidden_dim,
    #         activation=tf.nn.tanh
    #     )
    # )

    # add linear output layer
    layers.append(
        tf.keras.layers.Dense(
            units=state_dim,
            activation=None,
            use_bias=False
        )
    )

    encoder = tf.keras.Sequential(layers)

    return encoder


def make_posterior(h):

    h_mu = h[:,:,0:2]
    h_L = h[:,:,2:]

    qz = tfd.Independent(
        distribution=tfd.MultivariateNormalTriL(
            loc=h_mu,
            scale_tril=tfp.math.fill_triangular(h_L)
        ),
        reinterpreted_batch_ndims=2
    )

    return qz


DecoderParams = namedtuple(
    'DecoderParams',
    [
        'R',    # variance parameter of observation model
        'Q',    # variance parameter of dynamic model
        'L0',   # lower-tri scale matrix of variational posterior
        'mu0'   # mean of initial variational posterior
    ]
)


def decoder(codes, inputs, params, h):
    # codes are samples z, z' ~ q(z, z'|Y, X)

    assert(isinstance(codes, tf.Tensor))
    assert(isinstance(inputs, EncoderInputs))
    assert(isinstance(params, DecoderParams))
    assert(isinstance(h, tf.Tensor))

    def observation_density(codes, inputs, params):
        # p(y(t) | z(t), x(t)) = Normal(z(t)*x(t), R)
        py = tfd.Normal(loc=codes[:,:,:,0:1]*inputs.X, scale=tf.math.sqrt(params.R))
        py = tfd.Independent(py, reinterpreted_batch_ndims=3)

        return py

    def transition_density(codes, params):
        # p(z(t) | z(t-1)) = Normal(z(t-1), Q)
        pz = tfd.Normal(loc=codes[:,:,:,1:2], scale=tf.math.sqrt(params.Q))
        pz = tfd.Independent(pz, reinterpreted_batch_ndims=3)

        return pz

    def prior_density(params, h):
        # q(z(t-1) | y(t-1), x(t-1))
        # the marginalized variational posterior from the previous time step
        
        batch_size = h.shape[0]

        h_loc = h[:,:,0:2]
        h_scale_tril = tfp.math.fill_triangular(h[:,:,2:])    

        mu0 = params.mu0[0]*tf.ones([batch_size, 1, 1])
        L0 = params.L0[0,0]*tf.ones([batch_size, 1, 1])

        loc = tf.concat(
            [mu0, h_loc[:,0:-1,0:1]],
            axis=1 # concat on time axis
        )

        scale = tf.concat(
            [L0, h_scale_tril[:,0:-1,0:1,0]],
            axis=1 # concat on time axis
        )

        qzm = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) 
        qzm = tfd.Independent(qzm, reinterpreted_batch_ndims=2)

        return qzm

    py = observation_density(codes, inputs, params)
    pz = transition_density(codes, params)
    qzm = prior_density(params, h)
        

    return py, pz, qzm

@tf.function()
def loss_func(inputs, params, h, codes, num_samples):
    # approximates ELBO with monte-carlo samples from qz

    assert(isinstance(inputs, EncoderInputs))
    assert(isinstance(params, DecoderParams))
    assert(isinstance(h, tf.Tensor))
    assert(isinstance(codes, tf.Tensor))

    def mc_log_joint(inputs, params, h, codes):

        py, pz, qzm = decoder(codes, inputs, params, h)
        
        L1 = py.log_prob(inputs.Y)
        L2 = pz.log_prob(codes[:,:,:,0:1])
        L3 = qzm.log_prob(codes[:,:,:,1:2])

        LL = L1 + L2 + L3

        return LL

    log_probs = []
    log_probs.append(mc_log_joint(inputs, params, h, codes))

    qz = make_posterior(h)

    loss = -tf.reduce_sum(log_probs)/float(num_samples) - qz.entropy()

    return loss
    

def make_fake_data(batch_size, sequence_len, num_cycles, Q, R, z0, P0):

    xlist = []
    ylist = []
    zlist = []

    for ii in range(batch_size):
        phi = 2*np.pi*tf.range(sequence_len, dtype=tf.float32)/(sequence_len/num_cycles) + 2*tf.random.normal([1])
        x = tf.math.sin(phi)
        z = tf.scan(
            lambda acc, a: acc + a,
            tf.random.normal([sequence_len], mean=0, stddev=tf.math.sqrt(Q)),
            initializer=z0+tf.math.sqrt(P0)*tf.random.normal([])
        )
        y = z*x + tf.random.normal([sequence_len], mean=0, stddev=tf.math.sqrt(R))

        # expand batch dim
        x = tf.expand_dims(x, axis=0)
        y = tf.expand_dims(y, axis=0)
        z = tf.expand_dims(z, axis=0)

        xlist.append(x)
        ylist.append(y)
        zlist.append(z)

    xbatch = tf.concat(xlist, axis=0)
    ybatch = tf.concat(ylist, axis=0)
    zbatch = tf.concat(zlist, axis=0)

    return xbatch, ybatch, zbatch

    


if __name__ == '__main__':

    ''' 
    make fake data...

    phi(t) = 2*pi*t/42
    x(t) = sin(phi(t))
    z(t) = z(t-1) + w(t), w(t) ~ Normal(0, Q)
    y(t) = z(t)*x(t) + v(t), v(t) ~ Normal(0, R)

    '''

    batch_size = 40
    sequence_len = 200
    num_cycles = 3
    Q = 0.1
    R = 0.1
    z0 = 1.0
    P0 = 2.0

    xbatch, ybatch, zbatch = make_fake_data(batch_size, sequence_len, num_cycles, Q, R, z0, P0)



    # encoder to estimate shaping parameters of variational posterior
    mu0 = tf.zeros([2]) + z0
    L0 = tf.math.sqrt(P0)*tf.eye(2)
    triL_mask = tfp.math.fill_triangular(tf.ones([3], dtype=tf.bool))

    encoder = make_encoder(batch_size, sequence_len, hidden_dim=24)
    run_encoder = tf.function(encoder)

    initial_state = tf.concat(
        [
            tf.expand_dims(mu0, axis=0),
            tf.expand_dims(tf.boolean_mask(L0, triL_mask), axis=0)
        ],
        axis=-1
    )
    initial_state = tf.ones([batch_size, 1])*initial_state

    stacked_encoder_input = tf.concat(
        [
            tf.expand_dims(ybatch, axis=-1),
            tf.expand_dims(xbatch, axis=-1)
        ],
        axis=-1
    )

    inputs = EncoderInputs(
        Y=tf.expand_dims(ybatch, axis=-1),
        X=tf.expand_dims(xbatch, axis=-1)
    )
    
    decoder_params = DecoderParams(R=R, Q=Q, L0=L0, mu0=mu0)



    # set up training loop
    optimizer = tf.keras.optimizers.Adam(1e-3)

    num_steps = 10000
    num_samples = 25

    losses = []
    best_loss = 1e8


    @tf.function
    def train_body():

         # initialize encoder states
        encoder.layers[0].reset_states(states=[initial_state, initial_state])

        with tf.GradientTape() as g:
            g.watch(encoder.trainable_variables)
            
            # run encoder, sample codes
            h = run_encoder(stacked_encoder_input)
            qz = make_posterior(h)
            codes = qz.sample(num_samples)
            loss = loss_func(inputs, decoder_params, h, codes, num_samples)/float(batch_size*sequence_len)
            grads = g.gradient(loss, encoder.trainable_variables)

        return h, loss, grads


    # training loop
    for kk in range(num_steps):

        tt = time.time()
        h, loss, grads = train_body()
        dt = time.time() - tt
       
        if loss <= best_loss:
            hbest = h
            best_loss = loss

        print('iter: {}, time: {}, loss: {}'.format(kk, dt, loss))
        clipped_grads = [tf.clip_by_value(grad, -.001, .001) for grad in grads]
        # clipped_grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        optimizer.apply_gradients(zip(clipped_grads, encoder.trainable_variables))
        losses.append(loss)


    print('Done!')