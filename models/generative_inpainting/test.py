import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import os

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='/Users/kumarakahatapitiya/Desktop/codebase/generative_inpainting-master/my_test/val_image', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask_dir', default='/Users/kumarakahatapitiya/Desktop/codebase/generative_inpainting-master/my_test/val_mask', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output_dir', default='/Users/kumarakahatapitiya/Desktop/codebase/generative_inpainting-master/my_test/val_out', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='/Users/kumarakahatapitiya/Desktop/codebase/generative_inpainting-master/20180731133532416716_localhost.localdomain_ms_full_NORMAL_wgan_gp_ms_full_256_1', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    ng.get_gpus(1)
    #dummy='/Users/kumarakahatapitiya/Downloads/generative_inpainting-master/my_test/image/000011_image.png'
    args = parser.parse_args()

    model = InpaintCAModel()
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    output_dir = args.output_dir

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image=cv2.imread(os.path.join(image_dir,filename))
            mask=cv2.imread(os.path.join(mask_dir,filename))
            write_out=os.path.join(output_dir,filename)
            assert image.shape == mask.shape

            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            print('Shape of image: {}'.format(image.shape))

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)

            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            with tf.Session(config=sess_config) as sess:
                
                input_image = tf.constant(input_image, dtype=tf.float32)
                output = model.build_server_graph(input_image,reuse=tf.AUTO_REUSE)

                # load pretrained model
                vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                assign_ops = []
                for var in vars_list:
                    vname = var.name
                    from_name = vname
                    var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
                    assign_ops.append(tf.assign(var, var_value))
                sess.run(assign_ops)
                print('Model loaded.')

                output = (output + 1.) * 127.5
                output = tf.reverse(output, [-1])
                output = tf.saturate_cast(output, tf.uint8)
                result = sess.run(output)
                cv2.imwrite(write_out, result[0][:, :, ::-1])
