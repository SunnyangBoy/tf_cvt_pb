"""
此文件可以把ckpt模型转为pb模型
"""
import tensorflow as tf
import os
# from create_tf_record import *
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow


def get_model_filenames(model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)  # 通过checkpoint文件找到模型文件名
    if ckpt and ckpt.model_checkpoint_path:
        # ckpt.model_checkpoint_path表示模型存储的位置，不需要提供模型的名字，它会去查看checkpoint文件
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return ckpt_file


def freeze_graph(input_checkpoint, key):    # output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    output_node_names = key
    # output_node_names = "CRF_loss/transitions"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)   # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(","))
 
        with tf.gfile.GFile(output_graph, "wb") as f:
             f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))    # 得到当前图有几个操作节点


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model_exp = "/home/ubuntu/cs/tensorflow_pb/checkpoints/NER_ckpt"
    ckpt_file = get_model_filenames(model_exp)
    print('Checkpoint file:%s' % ckpt_file)
    reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(model_exp, ckpt_file))
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name:", key)

        input_checkpoint = '/home/ubuntu/cs/tensorflow_pb/checkpoints/NER_ckpt/-44550'
        # out_pb_path = '/home/ubuntu/cs/tensorflow_pb/checkpoints/NER_ckpt/NER_PbModel.pb'
        freeze_graph(input_checkpoint, key)  # out_pb_path)
