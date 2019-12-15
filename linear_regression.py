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

    # plt.plot(x)
    # plt.plot(z)
    # plt.plot(y)
    # plt.show()