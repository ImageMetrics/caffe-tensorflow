#!/usr/bin/env python

import argparse
import sys
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

def convert(model,output,shape):
    from alexnet import AlexNet as MyNet
    MyNet=getattr(__import__(model),model) 
    batch_size = 1
    data_node = tf.placeholder(tf.float32, shape)
    net = MyNet({'data': data_node})
    model_dir='./'
    with tf.Session() as sess:
        output_graph = sess._graph
        net.load(data_path=model+'.npy', session=sess)
        graph = convert_variables_to_constants(sess, sess.graph_def, [output])
        tf.train.write_graph(graph, '.', model+'.pb', as_text=False)

def main():
    input_height = 227
    input_width = 227
    input_channel=3
    input_batch=1
    model="LeNet"
    output="prob"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model name")
    parser.add_argument("--output", help="output name")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_channel", type=int, help="input channel")
    parser.add_argument("--input_batch", type=int, help="input batch")
    args = parser.parse_args()

    if args.input_height:
       input_height = args.input_height
    if args.input_width:
       input_width = args.input_width
    if args.input_channel:
       input_channel = args.input_channel
    if args.input_batch:
       input_batch = args.input_batch

    if args.model:
       model = args.model
    if args.output:
       output = args.output

    convert(model,output,(input_batch,input_height,input_width,input_channel))


if __name__ == '__main__':
    main()
