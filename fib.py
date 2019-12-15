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
    