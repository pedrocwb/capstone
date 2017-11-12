#!/usr/bin/env python
# pylint: disable=missing-docstring
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
import json
import logging
import numpy as np
import os.path
import sys

import scipy as scp
import scipy.misc

sys.path.insert(1, "incl/")

import tensorflow as tf

import tensorvision.utils as utils
import tensorvision.core as core
import tensorvision.analyze as ana
from seg_utils import seg_utils as seg

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

flags = tf.app.flags
FLAGS = flags.FLAGS

test_file = 'lung_data/testing.txt'


def create_test_output(hypes, sess, image_pl, softmax):
    data_dir = hypes['dirs']['data_dir']
    data_file = os.path.join(data_dir, test_file)
    image_dir = os.path.dirname(data_file)

    logdir = "test_images/"
    logdir_rb = "test_images_rb/"
    logdir_green = "test_images_green/"

    logging.info("Images will be written to {}/test_images_{{green, rg}}"
                 .format(logdir))

    logdir = os.path.join(hypes['dirs']['output_dir'], logdir)
    logdir_rb = os.path.join(hypes['dirs']['output_dir'], logdir_rb)
    logdir_green = os.path.join(hypes['dirs']['output_dir'], logdir_green)

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    if not os.path.exists(logdir_rb):
        os.mkdir(logdir_rb)

    if not os.path.exists(logdir_green):
        os.mkdir(logdir_green)

    thresh = np.array(range(0, 256))/255.0
    total_fp = np.zeros(thresh.shape)
    total_fn = np.zeros(thresh.shape)
    total_posnum = 0
    total_negnum = 0



    image_list = []

    with open(data_file) as file:
        for i, image_file in enumerate(file):
            print(image_file)
            real_image = image_file.split(" ")[0]        #image
            image_gt   = image_file.split(" ")[1]        #mask
            real_image = real_image.rstrip()
            image_gt   = image_gt.rstrip()

            real_image = os.path.join(image_dir, real_image)
            image_gt   = os.path.join(image_dir, image_gt)

            image = scp.misc.imread(real_image)
            gt_image    = scp.misc.imread(image_gt)
            image = image[:,:,:3]
            gt_image = gt_image[:,:,:3]
            shape = image.shape

            feed_dict = {image_pl: image}

            output = sess.run([softmax['softmax']], feed_dict=feed_dict)
            output_im = output[0][:, 1].reshape(shape[0], shape[1])

            ov_image = seg.make_overlay(image, output_im)
            hard = output_im > 0.5
            green_image = utils.fast_overlay(image, hard)

            name = os.path.basename(real_image)

            FN, FP, posNum, negNum = eval_image(hypes, gt_image, output_im)

            save_file = os.path.join(logdir, name)
            logging.info("Writing file: %s", save_file)
            scp.misc.imsave(save_file, output_im)

            save_file = os.path.join(logdir_rb, name)
            scp.misc.imsave(save_file, ov_image)

            save_file = os.path.join(logdir_green, name)
            scp.misc.imsave(save_file, green_image)

            total_fp += FP
            total_fn += FN
            total_posnum += posNum
            total_negnum += negNum


        eval_dict = seg.pxEval_maximizeFMeasure(
            total_posnum, total_negnum, total_fn, total_fp, thresh=thresh)

        logging.info(' MaxF1 : % 0.04f ' % (100*eval_dict['MaxF']))
        logging.info(' Average Precision : % 0.04f ' % (100*eval_dict['AvgPrec']))



def eval_image(hypes, gt_image, cnn_image):
    """."""
    thresh = np.array(range(0, 256))/255.0
    road_gt = gt_image[:, :, 2] > 0
    valid_gt = gt_image[:, :, 0] > 0

    FN, FP, posNum, negNum = seg.evalExp(road_gt, cnn_image,
                                         thresh, validMap=None,
                                         validArea=valid_gt)

    return FN, FP, posNum, negNum

def _create_input_placeholder():
    image_pl = tf.placeholder(tf.float32)
    label_pl = tf.placeholder(tf.float32)
    return image_pl, label_pl


def do_inference(logdir):
    """
    Analyze a trained model.

    This will load model files and weights found in logdir and run a basic
    analysis.

    Parameters
    ----------
    logdir : string
        Directory with logs.
    """
    hypes = utils.load_hypes_from_logdir(logdir)
    modules = utils.load_modules_from_logdir(logdir)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # prepaire the tv session

        with tf.name_scope('Validation'):
            image_pl, label_pl = _create_input_placeholder()
            image = tf.expand_dims(image_pl, 0)
            softmax = core.build_inference_graph(hypes, modules,
                                                 image=image)

        sess = tf.Session()
        saver = tf.train.Saver()

        core.load_weights(logdir, sess, saver)

        create_test_output(hypes, sess, image_pl, softmax)
    return


def main(_):
    """Run main function."""
    if FLAGS.logdir is None:
        logging.error("No logdir are given.")
        logging.error("Usage: tv-analyze --logdir dir")
        exit(1)

    if FLAGS.gpus is None:
        if 'TV_USE_GPUS' in os.environ:
            if os.environ['TV_USE_GPUS'] == 'force':
                logging.error('Please specify a GPU.')
                logging.error('Usage tv-train --gpus <ids>')
                exit(1)
            else:
                gpus = os.environ['TV_USE_GPUS']
                logging.info("GPUs are set to: %s", gpus)
                os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    else:
        logging.info("GPUs are set to: %s", FLAGS.gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

    utils.load_plugins()

    logdir = os.path.realpath(FLAGS.logdir)
    logging.info("Starting to analyze Model in: %s", logdir)
    do_inference(logdir)
    logging.info("Finished! ")


if __name__ == '__main__':
    tf.app.run()
