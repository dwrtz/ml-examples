# Table of Contents
- /Users/djwurtz/proj/ml-examples/.gitattributes
- /Users/djwurtz/proj/ml-examples/.vscode/settings.json
- /Users/djwurtz/proj/ml-examples/LICENSE
- /Users/djwurtz/proj/ml-examples/README.md
- /Users/djwurtz/proj/ml-examples/fib.py
- /Users/djwurtz/proj/ml-examples/issue.py
- /Users/djwurtz/proj/ml-examples/learning_linear_regression.py
- /Users/djwurtz/proj/ml-examples/linear_regression.py

## File: /Users/djwurtz/proj/ml-examples/.gitattributes

- Extension: .gitattributes
- Language: unknown
- Size: 66 bytes
- Created: 2026-04-24 08:34:37
- Modified: 2026-04-24 08:34:37

### Code

````unknown
# Auto detect text files and perform LF normalization
* text=auto

````

## File: /Users/djwurtz/proj/ml-examples/.vscode/settings.json

- Extension: .json
- Language: unknown
- Size: 85 bytes
- Created: 2026-04-24 08:34:37
- Modified: 2026-04-24 08:34:37

### Code

````unknown
{
    "python.pythonPath": "C:\\Users\\david\\env\\tf-nightly\\Scripts\\python.exe"
}
````

## File: /Users/djwurtz/proj/ml-examples/LICENSE

- Extension: 
- Language: unknown
- Size: 1068 bytes
- Created: 2026-04-24 08:34:37
- Modified: 2026-04-24 08:34:37

### Code

````unknown
MIT License

Copyright (c) 2019 David Wurtz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

````

## File: /Users/djwurtz/proj/ml-examples/README.md

- Extension: .md
- Language: unknown
- Size: 42 bytes
- Created: 2026-04-24 08:34:37
- Modified: 2026-04-24 08:34:37

### Code

````unknown
# ml-examples
 mostly tensorflow examples

````

## File: /Users/djwurtz/proj/ml-examples/fib.py

- Extension: .py
- Language: python
- Size: 891 bytes
- Created: 2026-04-24 08:34:37
- Modified: 2026-04-24 08:34:37

### Code

````python
import os
import tensorflow as tf
from collections import namedtuple

os.environ['CUDA_VISIBLE_DEVICES']= '-1'


def fib(N):

    FibState = namedtuple(
        'FibState',
        [
            'z1',
            'z2'
        ]
    )

    initial_state = FibState(
        z1=tf.dtypes.cast(0, tf.int64),
        z2=tf.dtypes.cast(1, tf.int64)
    )

    def inner_fib(current_state, elems):

        next_state = FibState(
            z1=current_state.z1 + current_state.z2,
            z2=current_state.z1
        )

        return next_state

    if N == 0:
        return 0

    else:
        result = tf.foldr(
            inner_fib, 
            tf.zeros(N, dtype=tf.int64), 
            initializer=initial_state
        )

        return result.z1.numpy()


if __name__ == '__main__':

    for n in range(100):
        print('fib({}) = {}'.format(n, fib(n)))

    print('Done!')
    
````

## File: /Users/djwurtz/proj/ml-examples/issue.py

- Extension: .py
- Language: python
- Size: 442 bytes
- Created: 2026-04-24 08:34:37
- Modified: 2026-04-24 08:34:37

### Code

````python
# RESOLVED: use tfd.MultivariateNormalDiag
import os
import tensorflow_probability as tfp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tfd = tfp.distributions


p = tfd.MultivariateNormalTriL(loc=[0,0], scale_tril=[[1,0],[0,-1]], validate_args=True)
q = tfd.Normal(loc=0, scale=-1)
r = tfd.MultivariateNormalDiag(loc=[0], scale_diag=[-1], validate_args=True)

print(p.log_prob([0,0]))
print(q.log_prob(0))
print(r.log_prob([0]))

print('Done!')
````

## File: /Users/djwurtz/proj/ml-examples/learning_linear_regression.py

- Extension: .py
- Language: python
- Size: 11136 bytes
- Created: 2026-04-24 08:34:37
- Modified: 2026-04-24 08:34:37

### Code

````python
import os, time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple
import tensorflow_probability as tfp
tfd = tfp.distributions

os.environ['CUDA_VISIBLE_DEVICES']= '0'

seed = 5678
tf.random.set_seed(seed)
np.random.seed(seed)



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
            units=hidden_dim,
            return_sequences=True,
            stateful=False,
            batch_input_shape=[batch_size, sequence_len, 2],
            unroll=unroll
        )
    )

    # # add second lstm layer
    # layers.append(
    #     tf.keras.layers.LSTM(
    #         units=hidden_dim,
    #         return_sequences=True,
    #         unroll=unroll
    #     )
    # )

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
    

def make_fake_data(batch_size, sequence_len, num_cycles_min_max, Q, R, z0, P0):

    xlist = []
    ylist = []
    zlist = []



    for ii in range(batch_size):
        num_cycles = np.random.randint(*num_cycles_min_max)
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
    num_cycles_min_max = [3, sequence_len//8]
    Q = 0.1
    R = 0.1
    z0 = 1.0
    P0 = 2.0

    xbatch, ybatch, zbatch = make_fake_data(batch_size, sequence_len, num_cycles_min_max, Q, R, z0, P0)



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
    num_samples = 50

    losses = []
    best_loss = 1e8


    @tf.function
    def train_body():

        # initialize encoder states
        # encoder.layers[0].reset_states(states=[initial_state, initial_state])

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
    grad_min = -.001
    grad_max = .001
    for kk in range(num_steps):

        tt = time.time()
        h, loss, grads = train_body()
        dt = time.time() - tt
       
        if loss <= best_loss:
            hbest = h
            best_loss = loss

        print('iter: {}, time: {}, loss: {}'.format(kk, dt, loss))
        clipped_grads = [tf.clip_by_value(grad, grad_min, grad_max) for grad in grads]
        # clipped_grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        optimizer.apply_gradients(zip(clipped_grads, encoder.trainable_variables))
        losses.append(loss)


    # plotting
    from linear_regression import KalmanFilterInputs, KalmanFilterStates, KalmanFilterParams, kalman_filter
    def plot_batch(idx):
        z_true = zbatch[idx,:]
        kf_inputs = KalmanFilterInputs(y=ybatch[idx,:], x=xbatch[idx,:])
        kf_initial_states = KalmanFilterStates(z=z0, P=P0)
        kf_params = KalmanFilterParams(R=R, Q=Q)
        kf_outputs = kalman_filter(kf_inputs, kf_initial_states, kf_params)

        kf_upper = kf_outputs.z + 2*tf.math.sqrt(kf_outputs.P)
        kf_lower = kf_outputs.z - 2*tf.math.sqrt(kf_outputs.P)


        h_mu = h[idx, :, 0:2]
        h_L = h[idx, :, 2:]
        h_tril = tfp.math.fill_triangular(h_L)

        net_upper = h_mu[:,0] + 2*tf.math.abs(h_tril[:,0,0])
        net_lower = h_mu[:,0] - 2*tf.math.abs(h_tril[:,0,0])


        # make plots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
        ax1.plot(xbatch[idx,:], color='blue', label='x')
        ax1.plot(ybatch[idx,:], color='green', label='y')
        ax1.legend()
        ax1.set_title('observations')
        ax1.grid(True)

        ax2.fill_between(np.arange(sequence_len), kf_upper, kf_lower, where=(kf_upper > kf_lower), alpha=0.3, color='red')
        ax2.plot(kf_outputs.z, color='red', label='kf z est.')
        ax2.plot(z_true, dashes=[1, 1], color='black', label='z true')
        ax2.fill_between(np.arange(sequence_len), net_upper, net_lower, where=(net_upper > net_lower), alpha=0.3, color='blue')
        ax2.plot(h_mu[:,0], color='blue', label='net z est.')
        ax2.legend()
        ax2.set_title('true and est. z with +/- 2-sigma interval')
        ax2.grid(True)


    print('Done!')
````

## File: /Users/djwurtz/proj/ml-examples/linear_regression.py

- Extension: .py
- Language: python
- Size: 2819 bytes
- Created: 2026-04-24 08:34:37
- Modified: 2026-04-24 08:34:37

### Code

````python
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple

os.environ['CUDA_VISIBLE_DEVICES']= '-1'


'''
This is a dynamic linear regression model...
p(y|z,x)    = Normal(z*x, R)
P(z|z')     = Normal(z', Q)

'''


KalmanFilterInputs = namedtuple(
    'KalmanFilterInputs',
    [
        'y',
        'x'
    ]
)

KalmanFilterStates = namedtuple(
    'KalmanFilterStates',
    [
        'z',
        'P'
    ]
)

KalmanFilterParams = namedtuple(
    'KalmanFilterParams',
    [
        'R',
        'Q'
    ]
)

def kalman_filter(inputs, initial_states, params):

    assert isinstance(inputs, KalmanFilterInputs)
    assert isinstance(initial_states, KalmanFilterStates)
    assert isinstance(params, KalmanFilterParams)

    def kalman_filter_inner(states, ins):

        # predict step
        z = states.z
        P = states.P + params.Q

        # update step
        e = ins.y - states.z*ins.x
        S = P*ins.x**2 + params.R
        K = P*ins.x/S

        z = z + K*e
        P = (1 - K*ins.x)*P

        next_states = KalmanFilterStates(
            z=z,
            P=P
        )

        return next_states

    result = tf.scan(
        kalman_filter_inner,
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


    # kalman filter to estimate p(z(t)|x(0..t),y(0..t))
    inputs = KalmanFilterInputs(y=y, x=x)
    initial_states = KalmanFilterStates(z=z0, P=1e1)
    params = KalmanFilterParams(R=R, Q=Q)

    outputs = kalman_filter(inputs, initial_states, params)
    
    upper = outputs.z + 2*tf.math.sqrt(outputs.P)
    lower = outputs.z - 2*tf.math.sqrt(outputs.P)


    # plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
    ax1.plot(x, color='blue', label='x')
    ax1.plot(y, color='green', label='y')
    ax1.legend()
    ax1.set_title('observations')
    ax1.grid(True)

    ax2.fill_between(np.arange(Npts), upper, lower, where=(upper > lower), alpha=0.3, color='red')
    ax2.plot(outputs.z, color='red', label='z est.')
    ax2.plot(z, dashes=[1, 1], color='black', label='z')
    ax2.legend()
    ax2.set_title('true and est. z with +/- 2-sigma interval')
    ax2.grid(True)
    
    plt.show()
````

