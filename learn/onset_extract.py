import cPickle as pickle
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from onset_net import OnsetNet
from util import *

tf.app.flags.DEFINE_string('data_txt_fp', '', 'Training dataset txt file with a list of pickled song files')
tf.app.flags.DEFINE_string('feats_dir', '', 'Subdirectory containing the audio features')
tf.app.flags.DEFINE_string('train_dir', '', 'Directory for temporary training files and model weights')
tf.app.flags.DEFINE_string('train_ckpt_fp', '', 'Path to model weights')
tf.app.flags.DEFINE_integer('context_radius', 7, 'Past and future context per training example')
tf.app.flags.DEFINE_integer('feat_dim', 80, 'Number of features per frame')
tf.app.flags.DEFINE_integer('nchannels', 3, 'Number of channels per frame')
tf.app.flags.DEFINE_bool('z_normalize_coeffs', False, 'Whether or not to normalize coeffs to zero mean, unit variance per band per channel.')
tf.app.flags.DEFINE_string('dense_layer_sizes', '256', 'Comma-separated list of dense layer sizes')
tf.app.flags.DEFINE_integer('export_feature_layer', 0, 'Which dense layer to use for features')
tf.app.flags.DEFINE_string('out_dir', '', 'Directory for output')

FLAGS = tf.app.flags.FLAGS
dtype = tf.float32

BATCH_SIZE = 512

def test():
    print 'Loading data...'
    with open(FLAGS.data_txt_fp, 'r') as f:
        pkl_fps = f.read().split()

    # Create model
    print 'Creating model'
    dense_layer_sizes = [int(x) for x in FLAGS.dense_layer_sizes.split(',')]
    model = OnsetNet(mode='eval',
                     batch_size=BATCH_SIZE,
                     audio_context_radius=FLAGS.context_radius,
                     audio_nbands=FLAGS.feat_dim,
                     audio_nchannels=FLAGS.nchannels,
                     nfeats=0,
                     cnn_filter_shapes=[(7,3,10),(3,3,20)],
                     cnn_init=tf.uniform_unit_scaling_initializer(factor=1.43, dtype=dtype),
                     cnn_pool=[(1,3),(1,3)],
                     cnn_rnn_zack=False,
                     rnn_cell_type=None,
                     rnn_size=0,
                     rnn_nlayers=0,
                     rnn_init=None,
                     rnn_nunroll=1,
                     rnn_keep_prob=1.0,
                     dnn_sizes=dense_layer_sizes,
                     dnn_init=tf.uniform_unit_scaling_initializer(factor=1.15, dtype=dtype),
                     dnn_keep_prob=1.0,
                     dnn_nonlin='relu',
                     target_weight_strategy='seq',
                     grad_clip=None,
                     opt=None,
                     export_feat_name='dnn_{}'.format(FLAGS.export_feature_layer))

    if FLAGS.z_normalize_coeffs:
        print 'Normalizing data to zero mean unit var'
        with open(os.path.join(FLAGS.train_dir, 'test_mean_std.pkl'), 'rb') as f:
            mean_per_band, std_per_band = pickle.load(f)

    if not os.path.isdir(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)

    with tf.Session() as sess:
        print 'Restoring model weights from {}'.format(FLAGS.train_ckpt_fp)
        model_saver = tf.train.Saver(tf.global_variables())
        model_saver.restore(sess, FLAGS.train_ckpt_fp)

        for pkl_fp in tqdm(pkl_fps):
            with open(os.path.join(FLAGS.feats_dir, pkl_fp), 'rb') as f:
                song_features = pickle.load(f)

            nframes = song_features.shape[0]

            # Normalize data
            if FLAGS.z_normalize_coeffs:
                apply_z_norm([(None, song_features, None)], mean_per_band, std_per_band)

            song_context, _ = model.prepare_train_batch([(None, song_features, None)], randomize_charts=False)
            song_export = []
            for i in xrange(0, nframes, BATCH_SIZE):
                batch_features = song_context[0][i:i + BATCH_SIZE]
                feed_dict = {
                    model.feats_audio: batch_features,
                    model.feats_other: np.zeros((batch_features.shape[0], 1, 0), dtype=np.float32)
                }
                batch_export = sess.run(model.feats_export, feed_dict=feed_dict)
                song_export.append(batch_export)
            song_export = np.concatenate(song_export)

            out_pkl_name = os.path.split(pkl_fp)[1]
            out_pkl_fp = os.path.join(FLAGS.out_dir, out_pkl_name)
            with open(out_pkl_fp, 'wb') as f:
                pickle.dump(song_export, f)


def main(_):
    test()


if __name__ == '__main__':
    tf.app.run()
