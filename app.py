from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
    
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from lib.model import data_loader, generator, SRGAN, test_data_loader, inference_data_loader, save_images, SRResnet
from lib.ops import *
import math
import time
import numpy as np

from flask import Flask, render_template, make_response
from flask import redirect, request, jsonify, url_for

import io
import os
import uuid
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

import base64

app = Flask(__name__)
app.secret_key = 's3cr3t'
app.debug = True
app._static_folder = os.path.abspath("templates/static/")

@app.route('/', methods=['GET'])
def index():
    title = 'Create the input'
    return render_template('layouts/index.html',
                           title=title)

'''
@app.route('/results/<uuid>', methods=['GET'])
def results(uuid):
    title = 'Result'
    data = get_file_content(uuid)
    return render_template('layouts/results.html',
                           title=title,
                           data=data)
'''

@app.route('/postmethod', methods = ['POST'])
def post_javascript_data():
    print("HEY THERE")
    jsdata = request.form['canvas_data'][23:]
    img_data = base64.b64decode(jsdata)

    '''
    encoded_data = jsdata.split(',')[1]
    nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    '''
    path = 'temp/' + get_random_hash(16)
    
    os.mkdir(path)
    
    with open(path + '/LR.png', 'wb+') as file:
        file.write(img_data)
    
    result_data = run_inference(path, jsdata)

    import shutil
    shutil.rmtree(path)
    
    params = { 'data' : result_data }
    return jsonify(params)

@app.route('/plot/<imgdata>')
def plot(imgdata):
    data = [float(i) for i in imgdata.strip('[]').split(',')]
    data = np.reshape(data, (200, 200))
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.axis('off')
    axis.imshow(data, interpolation='nearest')
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response

def create_csv(text):
    unique_id = str(uuid.uuid4())
    with open('images/'+unique_id+'.csv', 'a') as file:
        file.write(text[1:-1]+"\n")
    return unique_id

def get_file_content(uuid):
    with open('images/'+uuid+'.csv', 'r') as file:
        return file.read()




def get_random_hash(k=16):
    import random, string
    return(''.join(random.choices(string.ascii_letters + string.digits, k=k)))

def run_inference(path_to_image, data):

    Flags = tf.app.flags

    print("OUTPUT DIR")
    print(tf.flags.FLAGS.__flags)
    
    if(tf.flags.FLAGS.__flags == {}):
        # The system parameter
        Flags.DEFINE_string('output_dir', './result/', 'The output directory of the checkpoint')
        Flags.DEFINE_string('summary_dir', './result/log/', 'The dirctory to output the summary')
        Flags.DEFINE_string('mode', 'inference', 'The mode of the model train, test.')
        Flags.DEFINE_string('checkpoint', './SRGAN_pre-trained/model-200000', 'If provided, the weight will be restored from the provided checkpoint')
        Flags.DEFINE_boolean('pre_trained_model', True, 'If set True, the weight will be loaded but the global_step will still '
                             'be 0. If set False, you are going to continue the training. That is, '
                             'the global_step will be initiallized from the checkpoint, too')
        Flags.DEFINE_string('pre_trained_model_type', 'SRGAN', 'The type of pretrained model (SRGAN or SRResnet)')
        Flags.DEFINE_boolean('is_training', False, 'Training => True, Testing => False')
        Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')
        Flags.DEFINE_string('task', 'SRGAN', 'The task: SRGAN, SRResnet')
        # The data preparing operation
        Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch')
        Flags.DEFINE_string('input_dir_LR', path_to_image, 'The directory of the input resolution input data')
        Flags.DEFINE_string('input_dir_HR', None, 'The directory of the high resolution input data')
        Flags.DEFINE_boolean('flip', False, 'Whether random flip data augmentation is applied')
        Flags.DEFINE_boolean('random_crop', False, 'Whether perform the random crop')
        Flags.DEFINE_integer('crop_size', 24, 'The crop size of the training image')
        Flags.DEFINE_integer('name_queue_capacity', 2048, 'The capacity of the filename queue (suggest large to ensure'
                             'enough random shuffle.')
        Flags.DEFINE_integer('image_queue_capacity', 2048, 'The capacity of the image queue (suggest large to ensure'
                             'enough random shuffle')
        Flags.DEFINE_integer('queue_thread', 10, 'The threads of the queue (More threads can speedup the training process.')
        # Generator configuration
        Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')
        # The content loss parameter
        Flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature used in perceptual loss')
        Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
        Flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss')
        Flags.DEFINE_float('vgg_scaling', 0.0061, 'The scaling factor for the perceptual loss if using vgg perceptual loss')
        # The training parameters
        Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
        Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
        Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
        Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
        Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
        Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
        Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
        Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
        Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
        Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')

    FLAGS = Flags.FLAGS
    FLAGS.input_dir_LR = path_to_image

    # Check the output_dir is given
    if FLAGS.output_dir is None:
        raise ValueError('The output directory is needed')

    # Check the output directory to save the checkpoint
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    # Check the summary directory to save the event
    if not os.path.exists(FLAGS.summary_dir):
        os.mkdir(FLAGS.summary_dir)

    if FLAGS.mode == 'inference':
        # Check the checkpoint
        if FLAGS.checkpoint is None:
            raise ValueError('The checkpoint file is needed to performing the test.')

        # Declare the test data reader
        inference_data = inference_data_loader(FLAGS)

    inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
    path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')

    with tf.variable_scope('generator'):
        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
            gen_output = generator(inputs_raw, 3, reuse=tf.AUTO_REUSE, FLAGS=FLAGS)
        else:
            raise NotImplementedError('Unknown task!!')

    print('Finish building the network')

    with tf.name_scope('convert_image'):
        # Deprocess the images outputed from the model
        inputs = deprocessLR(inputs_raw)
        outputs = deprocess(gen_output)
        
        # Convert back to uint8
        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

        with tf.name_scope('encode_image'):
            save_fetch = {
                "path_LR": path_LR,
                "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
                "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
        }

    # Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    weight_initiallizer = tf.train.Saver(var_list)
    
    # Define the initialization operation
    init_op = tf.global_variables_initializer()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)
        
        max_iter = len(inference_data.inputs)
        print('Evaluation starts!!')

        input_im = np.array([inference_data.inputs[0]]).astype(np.float32)
        path_lr = inference_data.paths_LR[0]
        results = sess.run(save_fetch, feed_dict={inputs_raw: input_im, path_LR: path_lr})
            
        print("OUTPUTS")

        return str(base64.encodebytes(results["outputs"][0]),"utf-8").strip()

        # import base64
            
        # with open("./base_" + str(i) + ".txt", "wb") as f:
        #     f.write(base64.encodestring(results["outputs"][0]))
            
        # filesets = save_images(results, FLAGS)

        # for i, f in enumerate(filesets):
        #     print('evaluate image', f['name'])
    
    '''
    # Define the initialization operation
    init_op = tf.global_variables_initializer()
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)
        
        max_iter = len(inference_data.inputs)
        
        import base64
        /Users/egrigokhan/Documents/SRGAN/templates/static/js/script.js
        print('Evaluation starts!!')
        b = base64.b64decode(data)

        input_im = preprocess_test("./pic/images/img_001-inputs.png")
        path_lr = FLAGS.input_dir_LR
        input_im = input_im.reshape(1, input_im.shape[0], input_im.shape[1], input_im.shape[2])
        results = sess.run(save_fetch, feed_dict={inputs_raw: input_im, path_LR: path_lr})
        
        print("RESULTS")
        
        results = results["outputs"][0]
        
        with open('images/received.png', 'wb') as file:
            file.write(results)
            
        return str(base64.encodebytes(results),"utf-8").strip()
            
        # filesets = save_images(results, FLAGS)
        #     for i, f in enumerate(filesets):
        #     print('evaluate image', f['name'])
        '''
