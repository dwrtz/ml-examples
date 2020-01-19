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