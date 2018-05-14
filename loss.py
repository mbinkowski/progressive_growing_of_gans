# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import tfutil
import mmd

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

def G_wgan_acgan(G, D, opt, training_set, minibatch_size,
    cond_weight = 1.0): # Weight of the conditioning term.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
        loss += label_penalty_fakes * cond_weight
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    return loss

#----------------------------------------------------------------------------


def MMD_GAN_loss(G, D, opt, training_set, minibatch_size, reals, labels,
    kernel_type     = 'rbf',
    hessian_scale   = True,
    grad_expectation= 0,
    scale_variant   = 2,
    d_is_injective  = False,
    hs_lambda       = 10.0):

    optim_name = 'wmmd_gan' if hessian_scale else 'mmd_gan'
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)

    real = fp32(D.get_output_for(reals, is_training=True))
    fake = fp32(D.get_output_for(fake_images_out, is_training=True))
#    print('MMD LOSS: len(real, fake): ', len(real), len(fake)) 
    real_out, real_labels_out, real_grad = real
    fake_out, fake_labels_out, fake_grad = fake
    assert (real_grad is not None) & (fake_grad is not None)
#    print('MMD LOSS run(real_in): ', tfutil.run(real_in))

#    real_out = tfutil.autosummary('Loss/real_scores', real_out)
#    fake_out = tfutil.autosummary('Loss/fake_scores', fake_out)

    kernel_func = getattr(mmd, '_%s_kernel' % kernel_type)
    kernel_val = kernel_func(fake_out, real_out)
            
    with tf.variable_scope('loss'):
        unscaled_g_loss = mmd.mmd2(kernel_val, m=minibatch_size, n=minibatch_size)
        if hessian_scale:
            scale, norm_D, avg_norm2_jac = scale_by_hs_norm(minibatch_size, real_out, real_grad, fake_out, fake_grad, 
                grad_expectation, scale_variant, d_is_injective, hs_lambda)
            tf.summary.scalar(optim_name + '_unscaled_G', unscaled_g_loss)
            print('[*] Adding scale variant %d', scale_variant)
            tf.summary.scalar('dx_scale', avg_norm2_jac)
            print('[*] Hessian Scaling added')
        else:
            scale = 1.
        g_loss = unscaled_g_loss / scale
        tf.summary.scalar(optim_name + '_G', g_loss)
        tf.summary.scalar(optim_name + '_D', -g_loss)
        tf.summary.scalar(optim_name + '_norm_D', norm_D)
    
    return g_loss


G_mmdgan = MMD_GAN_loss


def D_mmdgan(G, D, opt, training_set, minibatch_size, reals, labels,
    kernel_type     = 'rbf',
    hessian_scale   = True,
    grad_expectation= 0,
    scale_variant   = 2,
    d_is_injective  = False,
    hs_lambda       = 10.0):
    return -MMD_GAN_loss(G, D, opt, training_set, minibatch_size, reals, labels,
                         kernel_type=kernel_type, hessian_scale=hessian_scale,
                         grad_expectation=grad_expectation, scale_variant=scale_variant,
                         d_is_injective=d_is_injective, hs_lambda=hs_lambda)


def scale_by_hs_norm(batch_size, real, real_grad, fake, fake_grad, grad_expectation, scale_variant, d_is_injective, hs_lambda):

    bs = batch_size
    if grad_expectation == 0:
        x_hat = real[:bs]
        x_grad = real_grad[:bs]
    elif grad_expectation == 1:
        x_hat = fake[:bs]
        x_grad = fake_grad[:bs]
    elif grad_expectation == 2:
        raise NotImplementedError('grad_expectation type 2 not available')
        alpha = tf.random_uniform(shape=[bs, 1, 1, 1])
        x_hat_data =  (1. - alpha) * self.images[:bs] + alpha * self.G[:bs]
    elif grad_expectation == 3:
        alpha = tf.random_uniform(shape=[bs, 1, 1, 1])
        x_hat = tf.boolean_mask(real[:bs],(alpha<0.5))  + tf.boolean_mask(fake[:bs],(alpha>=0.5))
        x_grad = tf.boolean_mask(real_grad[:bs],(alpha<0.5))  + tf.boolean_mask(fake_grad[:bs],(alpha>=0.5))
    
    x_hat = tf.stop_gradient(x_hat)
    if d_is_injective:
        raise NotImplementedError('injective d not avialable')
#        norme2_jac = squared_norm_jacobian(x_hat[:,:-self.input_dim ], x_hat_data)
#        avg_norm2_jac = tf.reduce_mean( norme2_jac  + tf.square(self.discriminator.scale_id_layer)*self.input_dim )
    else:
        avg_norm2_jac = tf.reduce_mean( x_grad )
    norm_discriminator = tf.reduce_mean(tf.square( x_hat ))

    if scale_variant == 0:
        epsilon = 0.00000001
        scale= (hs_lambda*avg_norm2_jac + epsilon)
    elif scale_variant == 2:
        scale= 1./(hs_lambda*avg_norm2_jac+1.)
    elif scale_variant == 3:
        scale= 1./(tf.maximum(hs_lambda*avg_norm2_jac+1, 4.))
    elif scale_variant == 4:
        epsilon = 0.00000001
        scale= 1+1./(hs_lambda*avg_norm2_jac+epsilon)
    elif scale_variant == 6:
        scale= 1+1./(hs_lambda*avg_norm2_jac+1)
    elif scale_variant == 8:
        scale= 1./(hs_lambda*(avg_norm2_jac + norm_discriminator) +1)
    return scale, norm_discriminator, avg_norm2_jac

