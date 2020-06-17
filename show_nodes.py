import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

model_exp = "/home/ubuntu/cs/tensorflow_pb/checkpoints/NER_ckpt"


def get_model_filenames(model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)  # 通过checkpoint文件找到模型文件名
    if ckpt and ckpt.model_checkpoint_path:
        # ckpt.model_checkpoint_path表示模型存储的位置，不需要提供模型的名字，它会去查看checkpoint文件
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return ckpt_file


if __name__ == '__main__':
    ckpt_file = get_model_filenames(model_exp)

    print('Checkpoint file: %s' % ckpt_file)
    reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(model_exp, ckpt_file))
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
