import os.path
import shutil
import argparse
import time
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import helper
import warnings
from distutils.version import LooseVersion
#import scipy.misc

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, graph, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param graph: Graph to use
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    with tf.name_scope('vgg16_encoder'):
    
        # get save model graph
        tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
        graph = tf.get_default_graph()                                      
        
        # load individual layer weights
        image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    #return layer weights
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


def get_regularization_loss(name='weights:0'):
    regularizers = 0
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if v.name.endswith(name):
            weight = tf.get_default_graph().get_tensor_by_name(v.name)
            regularizers += tf.nn.l2_loss(weight)
    return regularizers


def create_decoder(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Note: the fully connected layers of the VGG have been
    # decapitated and need to be replaced by a 1x1 convolution
    # to preserve spacial information
    #
    # Preform one by one covolution on the last vgg layer
    with tf.name_scope('decoder'):
        conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1,
                                    strides=(1, 1), padding='SAME',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                    name='conv1X1')
    
        # Upsample the convolution layer.
        up4 = tf.layers.conv2d_transpose(conv_1x1, 512,
                                         kernel_size=4, strides=2, padding='SAME',
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                         name='up4')

        # Skip layer connection--add layer 4 to upsampled output
        skip4 = tf.add(up4, vgg_layer4_out, name='skip4')
    
        # Upsample again
        up3 = tf.layers.conv2d_transpose(skip4, 256,
                                             kernel_size=4, strides=2, padding='SAME',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                             name='up3')
    
        # Skip layer connection--add layer 3 to upsampled output
        skip3 = tf.add(up3, vgg_layer3_out, name='skip3')
    
        # Upsample
        nn_last_layer = tf.layers.conv2d_transpose(skip3, num_classes, kernel_size=16,
                                                   strides=8, padding='SAME',
                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                                   name='output')
   
    return nn_last_layer


def create_predictions(nn_last_layer, num_classes):
    """
    Build the TensorFLow predictions.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param num_classes: Number of classes to classify
    :return: Tuple of (predictions, logits, prediction_softmax, prediction_class)
    """
    with tf.name_scope("predictions"):
        logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
        prediction_softmax = tf.nn.softmax(logits, name="prediction_softmax")
        prediction_class = tf.cast(tf.greater(prediction_softmax, 0.5), dtype=tf.float32, name='prediction_class')
    
    return logits, prediction_softmax, prediction_class

def create_optimizer(args, logits, correct_label, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param args: execution argument list
    :param predictions: TF Tensor of the predictions
    :param correct_label: TF of the correct label image
    :return: Tuple of (train_op, cross_entropy_loss, label_class)
    """
    LEARNING_RATE = args.learning_rate
    
    label_class = tf.reshape(correct_label, (-1, num_classes), name='label_class')
    
    with tf.name_scope("loss_op"):
        with tf.name_scope('cross_entropy'):
            softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                     labels=label_class)
        with tf.name_scope('regularization'):
            regularizers = get_regularization_loss()
            tf.summary.scalar('regularization',regularizers)
        with tf.name_scope('loss'):
            cross_entropy_loss = tf.reduce_mean(softmax_cross_entropy+.001*regularizers,
                                                name='cross_entropy_loss')
            tf.summary.scalar('loss', cross_entropy_loss)
 
    with tf.name_scope('learing_rate'):
        with tf.name_scope('global_step'):
            global_step = tf.Variable(0, trainable=False, name='global_step')
            tf.summary.scalar('global_step', global_step)
        with tf.name_scope('learning_rate'):
            rate = tf.train.natural_exp_decay(LEARNING_RATE, global_step, 250, 0.1)
            tf.summary.scalar('learing_rate', rate)
    
    with tf.name_scope('train_op'):
        optimizer = tf.train.AdamOptimizer(rate)
        train_op = optimizer.minimize(cross_entropy_loss)
    
    return train_op, cross_entropy_loss, label_class


def create_accuracy(args, prediction_class, label_class):
    """
    Build the TensorFLow accuracy operations.
    :param args: execution argument list
    :param args: execution line arguments
    :param prediction_class: TF place holder of the prediction class
    :return: Tuple of (accuracy, accuracy_op)
    """
    name = 'accuracy'
    with tf.name_scope(name):
        accuracy, accuracy_op = tf.metrics.accuracy(label_class,
                                                    prediction_class,
                                                    name=name)
            
    return accuracy, accuracy_op, name


def create_iou(args, prediction_class, label_class, num_labels):
    """
    Build the TensorFLow IoU operations.
    :param args: execution line arguments
    :param prediction_class: TF place holder of the prediction class
    :param label_class: TF place holder of the label class
    :return: Tuple of (iou, iou_op)
    """
    name = 'iou'
    with tf.name_scope(name):
        iou, iou_op = tf.metrics.mean_iou(label_class,
                                          prediction_class,
                                          num_labels,
                                          name=name)

    return iou, iou_op, name

def create_reset_running_vars(names):
    """
    Build the running variable tensor.
    :param names: list of scope names to search for
    : return: List of running variables
    """
    running_vars = []
    for name in names:
        running_vars += tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                         scope=name)
        
    return running_vars


def get_check_point_file(sess, args, saver):
    """
    Prepare/read check point file
    :param sess: TF Session
    :param args: execution line arguments
    :param saver: instantated saver class
    :return: void
    """
    CHECKPOINT_DIR = args.model_dir+'checkpoints/'
     
    # Create string of checkpoint directory
    chkpt_dir = os.path.dirname(CHECKPOINT_DIR)
    
    # Does checkpoint directory exist
    if os.path.exists(chkpt_dir):
        # Checkpoint directory exists
        # Get last checkpoint file in checkpoint directory
        last_checkpoint = tf.train.latest_checkpoint(chkpt_dir)
        if last_checkpoint:
            # Last checkpoint exists. Restore the checkpoint file
            print("restore checkpoint from {}".format(chkpt_dir))
            last_checkpoint = tf.train.latest_checkpoint(chkpt_dir)
            print("last checkpoint is {}".format(last_checkpoint))
            saver.restore(sess, last_checkpoint)
        else:
            # Remove existing checkpoints files
            # must first remove the directory then make the directory
            print("Removing exising checkpoints from {}".format(chkpt_dir))
            shutil.rmtree(chkpt_dir)
            os.makedirs(chkpt_dir)
    else:
        # Create checkpoint directory
        print("Make checkpoint directory {}".format(chkpt_dir))
        os.makedirs(chkpt_dir)
    
    return
        

def train_nn(sess, args, get_batches_fn, train_op, cross_entropy_loss,
             accuracy, accuracy_op, iou, iou_op, running_vars, 
             input_image, correct_label, keep_prob):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param args: execution line arguments
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param accuracy: accuracy
    :param accuracy_op: TF operation to update the accuracy per batch
    :param iou: itersection over union
    :param iou_op: TF Operation to update the iou per batch
    :param running_vars: TF tensor of list of running variables
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :return: void
    """

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    KEEP_PROB = args.dropout
    TENSORFLOW_DIR = args.tensorflow_dir
    CHECKPOINT_DIR = args.model_dir+'checkpoints/'
    CHECKPOINT_MDL = args.model_name
    SKIP_STEP = args.skip_step
    
    chk_pt_file = os.path.join(CHECKPOINT_DIR, CHECKPOINT_MDL)
    
    with tf.name_scope("Train_NN"):
        # Create tf saver
        saver = tf.train.Saver()
            
        # Load check point file (if it exists)
        get_check_point_file(sess, args, saver)
        
        # Instantiate a SummaryWriter to output summaries and the Graph.
        train_writer = tf.summary.FileWriter(TENSORFLOW_DIR+'/train', sess.graph)
        
        # Build the summary Tensor based on the TF collection of Summaries.
        merged = tf.summary.merge_all()
            
        print("Training...")
        print()
        
        f = open('Accuracy.txt','w')
    
        start_time = time.time()    
        
        summary_idx = 0
        
        # Define initializer to initialize/reset running variables
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)
        
        for epoch in range(EPOCHS):
            # Reset the running variables
            sess.run(running_vars_initializer)
            
            # Reset total loss and batch step counter
            total_loss = 0.0
            batch_step = 0
            for image, label in get_batches_fn(BATCH_SIZE):
                
                # Run session for the given feed dictionary
                feed_dict = {input_image: image,
                             correct_label: label,
                             keep_prob: KEEP_PROB}
                _, batch_loss, _, _, summaries = sess.run([train_op,
                                                          cross_entropy_loss,
                                                          accuracy_op,
                                                          iou_op,
                                                          merged], feed_dict=feed_dict)
                
                total_loss += batch_loss
                print("For epoch {} batch {} the batch loss is {}"
                      .format(epoch+1,batch_step+1,batch_loss))
                
                if( (summary_idx % SKIP_STEP) == 0):
                    train_writer.add_summary(summaries, summary_idx)
                    train_writer.flush()
                    
                batch_step += 1
                summary_idx += 1
   
            # Obtain the overall accuracy and iou
            epoch_accuracy, epoch_iou = sess.run([accuracy, iou])
            
            # Save the checkpoint
            saver.save(sess, chk_pt_file, epoch)
            
            print("EPOCH {} ...".format(epoch+1))
            print("Training loss = {:.5f}, accuracy = {:.5f}, iou = {:.5f}"
                  .format(total_loss, epoch_accuracy, epoch_iou))
            print()
            f.write("{} {:.5f} {:.5f}\n".format(epoch, epoch_accuracy, epoch_iou))
        
        # Close files
        f.close()
        train_writer.close()

        # Print total time to complete run
        print("Total time: {0} seconds".format(time.time() - start_time))
        
    return


def train(args, num_classes, image_shape, data_dir):
    """
    Create and train the model
    :param args: execution line arguments
    :num_classes: number of classes to train for
    :image_shape: input image shape
    :data_dir: directory that holes the training data
    :return: void
    """
    # Download pretrained vgg model
    print("Download pretrained model (if not already downloaded)")
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Clear default graph (fixes issues with python spyder)
    tf.reset_default_graph()
    
    # Use the default graph
    with tf.Graph().as_default() as graph:

        with tf.Session(graph=graph) as sess:            
            # Create function to get batches
            get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

            # OPTIONAL: Augment Images for better results
            #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
            
            # Path to vgg model
            vgg_path = os.path.join(data_dir, 'vgg')
            print("Vgg path is {}".format(vgg_path))
            
            # Load vgg_16 encoder layers
            image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess,
                                                                                  graph,
                                                                                  vgg_path)
            print("Loaded vgg_16 model")
            
            # Load the decoder layers
            nn_last_layer = create_decoder(layer3_out, layer4_out, layer7_out, num_classes)
            print("Constructed the decoder")
                    
            # create place holder for the correct_label variable
            correct_label = tf.placeholder(tf.float32, (None), name='correct_label')
            
            # Load predictions funciton
            logits, prediction_softmax, prediction_class = create_predictions(nn_last_layer, num_classes)
            print("Constructed the predictions")
            
            # Load the optimize funciton
            train_op, cross_entropy_loss, label_class = create_optimizer(args,
                                                                         logits,
                                                                         correct_label,
                                                                         num_classes)
            names = []
            print("Constructed the optimizer")
            
            # Load accuracy function
            accuracy, accurach_op, name = create_accuracy(args, prediction_class,
                                                  label_class)
            names.append(name)
            print("Constructed batch accuracy")
            
            # Load iou function
            iou, iou_op, name = create_iou(args, prediction_class, label_class,
                                     num_classes)
            names.append(name)
            print("Constructed batch iou")
            
            running_vars = create_reset_running_vars(names)
            
            # Tell TensorFlow that the model will be built into the default Graph.
            sess.run(tf.global_variables_initializer())
            
            # Initialize local variables for this to run
            sess.run(tf.local_variables_initializer())
            
            # Train NN using the train_nn function
            print("Start train_nn")
            train_nn(sess, args, get_batches_fn, train_op, cross_entropy_loss,
                     accuracy, accurach_op, iou, iou_op, running_vars,
                     image_input, correct_label, keep_prob)
    
            # Write the trained graph to pb file
            # (Must write as a binary file to use freeze graph tool)
            model_dir = args.model_dir
            model_name = args.model_name+'.pb'
            tf.train.write_graph(sess.graph,model_dir,model_name,as_text=False)
    return


def create_frozen_graph(args, output_node_names):
    """
    Freeze the trained model
    :param args: execution line arguments
    :output_node_names: comma seperated string of tensor output names
    :return: void
    """
    MODEL_DIR = args.model_dir
    MODEL_NAME = args.model_name

    # Define freeze_graph input parameters

    input_graph_path = os.path.join(MODEL_DIR, MODEL_NAME+'.pb')
    if not tf.gfile.Exists(input_graph_path):
        raise AssertionError("Model file {} doesn't exist".format(input_graph_path))
        
    checkpoint_dir = os.path.join(MODEL_DIR,'checkpoints')
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    if not checkpoint_file:
        raise AssertionError("Checkpoint file {} doesn't exist".format(checkpoint_file))
        
        
    input_saver_def_path = ""
    input_binary = True
    restore_op_name = ""
    filename_tensor_name = ""
    output_frozen_graph_name = os.path.join(MODEL_DIR, 'frozen_'+MODEL_NAME+'.pb')
    clear_devices = True


    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_file, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")
    
    return


def create_optimized_graph(args, input_node_names, output_node_names):
    """
    Freeze the trained model
    :param args: execution line arguments
    :input_node_names: string list of input tensors names
    :output_node_names: string list of output tensors names
    :return: void
    """
    MODEL_DIR = args.model_dir
    MODEL_NAME = args.model_name

    # Define optimized graph input parameters
    output_frozen_graph_name = os.path.join(MODEL_DIR, 'frozen_'+MODEL_NAME+'.pb')
    if not tf.gfile.Exists(output_frozen_graph_name):
        raise AssertionError("Frozen graph file {} doesn't exist".format(output_frozen_graph_name))
    
    output_optimized_graph_name = os.path.join(MODEL_DIR, 'optimized_'+MODEL_NAME+'.pb')
    
    # Optimize for inference
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_frozen_graph_name, 'rb') as f:
        data = f.read()
        input_graph_def.ParseFromString(data)
    
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def,
            input_node_names, # an array of the input node(s)
            output_node_names, # an array of output nodes
            tf.float32.as_datatype_enum)

    # Save the optimized graph  
    f = tf.gfile.FastGFile(output_optimized_graph_name, 'wb')
    f.write(output_graph_def.SerializeToString())

    return


def load_graph(args, graph_type):
    """
    Load frozen graph of the trained model
    :param args: execution line arguments
    :param graph_type: frozen/optimized
    :return: graph
    """
    MODEL_DIR = args.model_dir
    MODEL_NAME = args.model_name
    graph_name = os.path.join(MODEL_DIR, graph_type+"_"+MODEL_NAME+".pb")
    
    if not tf.gfile.Exists(graph_name):
        raise AssertionError("Forzen file {} doesn't exist".format(graph_name))
    
    # Load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    print("")
    print("Loading graph {}".format(graph_name))
    print("")
    with tf.gfile.GFile(graph_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


def run(args):
    num_classes = 2
    image_shape = (160, 576)
    data_dir = args.data_dir
    runs_dir = args.runs_dir
    num_images = args.num_images
    
    # Loop through the desired actions
    if not args.actions:
        print("*******************")
        print("")
        print("No action specified: Quit")
        print("")
        print("*******************")
        return
    
    for action in args.actions:
    
        if(action == 'train'):
            train(args, num_classes, image_shape, data_dir)
            
        elif(action == 'freeze'):
            print("")
            print("Freezing graph for output names: ")
            # create comma seperate string of output nodes to freeze
            scope_name = "predictions"
            node_names = ["logits", "prediction_softmax",
                          "prediction_class"]
            output_node_names = ""
            for name in node_names:
                output_name = scope_name+"/"+name
                output_node_names = output_node_names+output_name+","
                print("   {}".format(output_name))
            output_node_names = output_node_names[:-1]
            print(output_node_names)
            
            # create the frozen graph
            create_frozen_graph(args, output_node_names)
            print("")
            
        elif(action == 'optimize'):
            print("")
            print("Optimizing graph for input and output names: ")
            input_node_names = ["image_input", "keep_prob"]
            # create list of output nodes to freeze
            scope_name = "predictions"
            node_names = ["logits", "prediction_softmax",
                          "prediction_class"]
            output_node_names = []
            for name in node_names:
                output_name = scope_name+"/"+name
                output_node_names.append(output_name)
                print("   {}".format(output_name))

            create_optimized_graph(args, input_node_names, output_node_names)
            
        elif(action == 'predict'):
            # get the optimized graph
            graph = load_graph(args, 'optimized')

            # make default graph
            G = tf.Graph()
            # prdict
            with tf.Session(graph=G) as sess:
                # import input and output tensors from loaded graph
                image_input, keep_prob, logits = tf.import_graph_def(graph,
                                              return_elements=['image_input:0',
                                                               'keep_prob:0',
                                                               'predictions/logits:0'])

                # Predict by inference road images
                helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                              logits, keep_prob, image_input,
                                              num_images)
                sess.close()
            
        elif(action == 'video'):
            
            if not args.video_in:
                raise AssertionError("Video input file not defined")
            if not args.video_out:
                raise AssertionError("Video output file not defined")              

            video_file_in = args.video_in+'.mp4'
            video_file_out = args.video_out+'.mp4'
            
            if not os.path.isfile(video_file_in):
                raise AssertionError("Video file {} doesn't exist".format(video_file_in))
                
            graph = load_graph(args, 'optimized')

            # make default graph
            G = tf.Graph()
            # prdict
            with tf.Session(graph=G) as sess:
                # import input and output tensors from loaded graph
                image_input, keep_prob, logits = tf.import_graph_def(graph,
                                              return_elements=['image_input:0',
                                                               'keep_prob:0',
                                                               'predictions/logits:0'])
            
                helper.save_inference_video(video_file_in, video_file_out, sess,
                                            image_shape, logits, keep_prob,
                                            image_input, args.frame_range)

                sess.close()
                
        else:
            print("Undefined action {}".format(action))
    
                # Save inference data using helper.save_inference_samples
                #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                #                              predictions, keep_prob, image_input)
    
            # OPTIONAL: Apply the trained model to a video
    return

def get_args():
    # create an instance of the argument parser
    parser = argparse.ArgumentParser()
    
    # get the arguements
    parser.add_argument('--actions',
                        help='Choice: train/predict/freeze/optimise/video',
                        type=str,
                        choices=['train', 'predict', 'freeze', 'optimize', 'video'],
                        action='append')    

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--skip_step', type=int, default=10,
                        help='Number of steps between checkpoint saves')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Keep probability for training dropout.')
    parser.add_argument('--model_dir', type=str, default='./LaneClassifer/',
                      help='Model log directory')
    parser.add_argument('--model_name', type=str, default='LaneClassifier',
                        help='Model name')
    parser.add_argument('--tensorflow_dir', type=str, default='./my_graph',
                        help='Tensorflow graph directory')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory with image inputs')
    parser.add_argument('--runs_dir', type=str, default='./runs',
                        help='Directory to store processed images')
    parser.add_argument('--num_images', type=int,
                        help='Number of inference images to process [Default processes all]')
    parser.add_argument('--video_in', type=str, default='',
                        help='Video input file')
    parser.add_argument('--video_out', type=str, default='',
                        help='Video output file')
    parser.add_argument('--frame_range', type=int, nargs=2,
                        help='list video start and stop frames to process [Default processes all]')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # get argument list
    args = get_args()    
    print("Input args are: {}".format(args))
    
    # execute
    run(args)
