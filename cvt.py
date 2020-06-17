import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


def freeze_graph(ckpt, output_graph):
    output_node_names = 'Bi-LSTM/Reshape_1'
    saver = tf.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(',')
        )
        with tf.gfile.GFile(output_graph, 'wb') as fw:
            fw.write(output_graph_def.SerializeToString())
        print('{} ops in the final graph.'.format(len(output_graph_def.node)))


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ckpt = '/home/ubuntu/cs/tensorflow_pb/checkpoints/NER_ckpt/-44550'
pb = '/home/ubuntu/cs/tensorflow_pb/checkpoints/NER_ckpt/NER_PbModel.pb'

if __name__ == '__main__':
    freeze_graph(ckpt, pb)
