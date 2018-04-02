import os, sys
sys.path.append(os.getcwd())

import time
import numpy as np
import tensorflow as tf

import tflib as lib

from spin import Model

# Define spin system and generate training dataset
geometry = (16,16)
T = 3.
samples = 2000

x = Model()
x.generate_system(geometry=geometry, T=T)
x.generate_ensemble(n_samples=samples)
dataset = x.ensemble.configuration
dataset = dataset.reshape(samples, -1)

# Define GAN parameters
MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 20000 # How many generator iterations to train for 
OUTPUT_DIM = 16*16 # Number of pixels
N = geometry[0] # Ising 2D Dim
EVAL_BATCHES = 40 # Number of batches per eval step
DATASET_SIZE = samples # Number of samples to generate for train dataset
Z_DIM = 2*2*4*DIM

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = spin.gan.tflib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = spin.gan.tflib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):

    if noise is None:
        noise = tf.random_normal([n_samples, Z_DIM])

    # [n_samples, Z_DIM] -> [n_samples, 2, 2, 4*DIM]
    output = tf.reshape(noise, [n_samples, 4*DIM, 2, 2])

    # [n_samples, 2, 2, 4*DIM] -> [n_samples, 4, 4, 2*DIM]    
    output = spin.gan.tflib.ops.deconv2d.Deconv2D('Generator.2', 4 * DIM, 2 * DIM, 5, output)
    if MODE == 'wgan':
        output = spin.gan.tflib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 2, 3], output)
    output = tf.nn.relu(output)

    # [n_samples, 4, 4, 2*DIM] -> [n_samples, 8, 8. DIM]
    output = spin.gan.tflib.ops.deconv2d.Deconv2D('Generator.3', 2 * DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = spin.gan.tflib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 2, 3], output)
    output = tf.nn.relu(output)

    # [n_samples, 8, 8, DIM] -> [n_samples, 16, 16, 1] 
    output = spin.gan.tflib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    # Rescale output [0, 1] -> [-1, 1]
    output = output*2-1
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):

    # [n_samples, OUTPUT_DIM] -> [n_samples, 16, 16, 1]
    output = tf.reshape(inputs, [-1, 1, N, N])

    # [n_samples, 16, 16, 1] -> [n_samples, 8, 8, DIM]
    output = spin.gan.tflib.ops.conv2d.Conv2D('Discriminator.1', 1, DIM, 4, output, stride=2)
    output = LeakyReLU(output)

    # [n_samples, 8, 8, DIM] -> [n_samples, 4, 4, 2*DIM]
    output = spin.gan.tflib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2 * DIM, 4, output, stride=2)
    if MODE == 'wgan':
        output = spin.gan.tflib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 2, 3], output)
    output = LeakyReLU(output)

    # [n_samples, 4, 4, 2*DIM] -> [n_samples, 2, 2, 4*DIM]
    output = spin.gan.tflib.ops.conv2d.Conv2D('Discriminator.3', 2 * DIM, 4 * DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = spin.gan.tflib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 2, 3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 2*2*4*DIM])
    output = spin.gan.tflib.ops.linear.Linear('Discriminator.Output', 2 * 2 * 4 * DIM, 1, output)

    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_real, 
        tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# Dataset iterator
def inf_train_gen():
    while True:
        yield dataset[np.random.randint(dataset.shape[0], size=BATCH_SIZE), :]

# Train loop
with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    gen = inf_train_gen()

    gen_costs = []
    disc_costs = []

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _, _gen_cost = session.run([gen_train_op, gen_cost])
            gen_costs.append(_gen_cost)

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data}
            )
            disc_costs.append(_disc_cost)
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)
            
        if iteration % 50 == 49:
          print("iteration %d" % iteration)
          print("gen_cost: %f" % np.mean(np.array(gen_costs)))
          print("disc_cost: %f" % np.mean(np.array(disc_costs)))
          gen_costs = []
          disc_costs = []


        # Code for evaluating fake data, waiting for similar code in spin to appear
        #if iteration % 2000 == 1999:
        #  print("Generating samples and calculating properties")
        #  fake_dataset = []
        #  for j in range(EVAL_BATCHES):
        #    fake_dataset.append(session.run(fake_data))
        #  fake_dataset = np.stack(fake_dataset).reshape((-1, OUTPUT_DIM))
        #  fake_dataset = np.sign(fake_dataset)
        #  np.savetxt(samples_filename, fake_dataset)
        #  te, tm, tsh, tss = evaluate_ising_data(T, N, fake_dataset)
        #  
        #  with open(gen_filename, "a") as f:
        #    writer = csv.writer(f)
        #    writer.writerow([te,tm,tsh,tss])
    
        #  print("Test Energy: %f True Energy: %f Diff: %f" % (te, e, te-e))
        #  print("Test Magnetization: %f True Magnetization %f Diff: %f" % (tm, m, tm-m))
        #  print("Test Specific Heat: %f True Specific Heat: %f Diff: %f" % (tsh, sh, tsh-sh))
        #  print("Test Susceptibility: %f True Susceptibility: %f Diff: %f" % (tss, ss, tss-ss))

