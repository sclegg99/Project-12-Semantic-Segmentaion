import re
import random
import numpy as np
import os.path
import scipy
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import imageio


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = []
    for vgg_file in vgg_files:
        if not os.path.exists(vgg_file):
            missing_vgg_files.append(vgg_file)

    if missing_vgg_files:
        # Clean vgg dir
        print("Missing vgg file(s) {}".format(missing_vgg_files))
        print("Reload pre-trained vgg model")
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_lane_*.png'))}
    
    # remove files from image path that are not in the label_paths dictionary
    index = 0
    poplist=[]
    for image_file in image_paths:
        if os.path.basename(image_file) not in label_paths.keys():
            poplist = [index] + poplist # add index to begining of list
        index += 1
    for index in poplist:
        image_paths.pop(index)
        
    background_color = np.array([255, 0, 0])

    random.shuffle(image_paths)
    
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)

    return get_batches_fn


def gen_composite_image(sess, logits, keep_prob, image_input, image, image_shape):
    """
    Generate a composit output image
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_input: TF Placeholder for the image placeholder
    :param image: PIL input image
    :param image_shape: Tuple - Shape of image
    :return: composite image
    """
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_input: [image]})
    
    temp = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (temp > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    
    return np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits,
                           keep_prob, image_input, num_images):
    """
    Generate test output using the test images
    :param runs_dir: directory to store inference results
    :param data_dir: directory containing test images
    :param sess: TF session
    :param image_shape: Tuple - Shape of training image
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_input: TF Placeholder for the image placeholder
    :param num_images: number of images to process
    :return: void
    """
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print('Saving test images to: {}'.format(output_dir))
    
    image_files = glob(os.path.join(data_dir, 'data_road/testing',
                                    'image_2', '*.png'))
    if num_images:
        image_files = image_files[:num_images]

    # Run NN on test images and save them to HD

    for image_file in image_files:
        print("Processing image {}".format(image_file))
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        street_im = gen_composite_image(sess, logits, keep_prob, image_input,
                                        image, image_shape)
        name = os.path.basename(image_file)
        scipy.misc.imsave(os.path.join(output_dir, name), street_im)
        
    return


def save_inference_video(video_file_in, video_file_out, sess, image_shape, logits,
                         keep_prob, image_input, frame_range):
    """
    Generate video output 
    :param video_file_in: name of input video file
    :param video_file_out: name of video file of inference
    :param sess: TF session
    :param image_shape: Tuple - Shape of training image
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_input: TF Placeholder for the image placeholder
    :param video_in: video input object
    :param frame_range: range of frames to read
    :return: void
    """
 
    # Video object for reading video file
    vidin = imageio.get_reader(video_file_in)
    
    # Video object for writing video file
    vidout = imageio.get_writer(video_file_out, '.mp4', 'I')

    # Get number of frames in input video file
    nframes = vidin.get_meta_data()['nframes']
    video_size = tuple(reversed(vidin.get_meta_data()['size']))
    
    # Frames to process
    if not frame_range:
        start = 0
        stop = nframes
    else:
        start = frame_range[0]
        stop = frame_range[1]
        
    frames = list(range(start,stop))
    
    for frame in frames:
        image = scipy.misc.imresize(vidin.get_data(frame), image_shape)
        print("Processing video frame {}".format(frame))
        
        street_im = gen_composite_image(sess, logits, keep_prob, image_input,
                                        image, image_shape)
        
        street_im = scipy.misc.imresize(street_im, video_size)
        vidout.append_data(street_im)

    vidin.close()    
    vidout.close()
    
    return

    