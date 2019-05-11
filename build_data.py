import tensorflow as tf
import random
import os

try:
    from os import scandir
except ImportError:
    # Python 2 polyfill module
    from scandir import scandir

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_dir', 'data/DIV2K/Y_train_ag',
                       'Input directory')
tf.flags.DEFINE_string('gt_dir', '../data/DIV2K/Y_train_ag',
                       'Ground truth directory')
tf.flags.DEFINE_string('output_file', '../data/tfrecords/train_y.tfrecords',
                       'Output tfrecords file')


def data_reader(input_dir, gt_dir, shuffle=True):
    """Read images from input_dir then shuffle them
    Args:
      input_dir: string, path of input dir, e.g., /path/to/dir
      gt_dir: string, path of gt dir, e.g., /path/to/dir
    Returns:
      file_paths: list of strings
      gt_paths: list of strings
    """
    file_paths = []
    gt_paths = []

    for img_file in scandir(input_dir):
        if img_file.name.endswith('.png') and img_file.is_file():
            file_paths.append(img_file.path)

    for img_file in scandir(gt_dir):
        if img_file.name.endswith('.png') and img_file.is_file():
            gt_paths.append(img_file.path)

    if len(file_paths) != len(gt_paths):
        raise Exception('File paths and Ground truth paths not corresponding. Length mismatch.')

    for i in range(len(gt_paths)):
        if file_paths[i].split('\\')[-1] != gt_paths[i].split('\\')[-1]:
            raise Exception('File paths and Ground truth paths not corresponding. Not corresponding files.')

    if shuffle:
        # Shuffle the ordering of all image files in order to guarantee
        # random ordering of the images with respect to label in the
        # saved TFRecord files. Make the randomization repeatable.
        shuffled_index = list(range(len(file_paths)))
        random.seed(12345)
        random.shuffle(shuffled_index)

        file_paths = [file_paths[i] for i in shuffled_index]
        gt_paths = [gt_paths[i] for i in shuffled_index]

    return file_paths, gt_paths


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer, gt_file_path, gt_image_buffer):
    """Build an Example proto for an example.
    Args:
      file_path: string , path to an image file, e.g., '/path/to/example.PNG'
      image_buffer: string, PNG encoding of RGB image
      file_path: string , path to a grounfd truth image file, e.g., '/path/to/example.PNG'
      image_buffer: string, PNG encoding of RGB ground truth image
    Returns:
      Example proto
    """
    file_name = file_path.split('/')[-1]
    gt_file_name = gt_file_path.split('/')[-1]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
        'image/encoded_image': _bytes_feature((image_buffer)),
        'image/gt_file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(gt_file_name))),
        'image/gt_encoded_image': _bytes_feature((gt_image_buffer))
    }))
    return example


def data_writer(input_dir, gt_dir,  output_file):
    """Write data to tfrecords
    """
    file_paths, gt_paths = data_reader(input_dir, gt_dir)

    # create tfrecords dir if not exists
    output_dir = os.path.dirname(output_file)
    try:
        os.makedirs(output_dir)
    except os.error as e:
        pass

    images_num = len(file_paths)

    # dump to tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    for i in range(len(file_paths)):
        file_path = file_paths[i]
        gt_path = gt_paths[i]

        with tf.gfile.FastGFile(file_path, 'rb') as f:
            image_data = f.read()

        with tf.gfile.FastGFile(gt_path, 'rb') as f:
            gt_data = f.read()

        example = _convert_to_example(file_path, image_data, gt_path, gt_data)
        writer.write(example.SerializeToString())

        if i % 100 == 0:
            print("Processed {}/{}.".format(i, images_num))
    print("Done.")
    writer.close()


def main(unused_argv):
    data_writer(FLAGS.input_dir, FLAGS.gt_dir, FLAGS.output_file)


if __name__ == '__main__':
    tf.app.run()
