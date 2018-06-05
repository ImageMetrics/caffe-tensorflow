#!/usr/bin/env python

import argparse
import json
import sys, traceback
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

sys.path.insert(0, './')

def convert(model,inputs,outputs,shape):
    MyNet=getattr(__import__(model),model)
    inputArgs = {}
    for input in inputs:
        inputArgs[input] = tf.placeholder(tf.float32, shape, name=input)
    net = MyNet(inputArgs)
    model_dir='./'
    with tf.Session() as sess:
        output_graph = sess._graph
        net.load(data_path=model+'.npy', session=sess)
        try:
          graph = convert_variables_to_constants(sess, sess.graph_def, outputs)
          tf.train.write_graph(graph, '.', model + '.pb', as_text=False)
        except Exception, e:
          traceback.print_exc()
          print 'Valid nodes are:'
          for node in sess.graph_def.node:
            print '  ' + node.name
          exit(-1)

def main():
    input_height = 227
    input_width = 227
    input_channel=3
    input_batch=1
    model="LeNet"
    inputs=["data"]
    outputs=["prob"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model name")
    parser.add_argument("--input", action='append', help="input name")
    parser.add_argument("--output", action='append', help="output name")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_channel", type=int, help="input channel")
    parser.add_argument('--input-list-path', help='Input layer names list file path')
    parser.add_argument('--input-shape-list-path', help='Input layer shape list file path')
    parser.add_argument('--output-list-path', help='Output layer names list file path')
    parser.add_argument("--input_batch", type=int, help="input batch")
    args = parser.parse_args()

    if args.input_list_path:
        input_list_str = open(args.input_list_path, 'rb').read()
        inputs = json.loads(input_list_str)
        inputs = [y for x in inputs for y in x] # https://coderwall.com/p/rcmaea/flatten-a-list-of-lists-in-one-line-in-python
    elif args.input:
        inputs = args.input

    if args.input_shape_list_path:
        input_shape_list_str = open(args.input_shape_list_path, 'rb').read()
        input_shape_list = json.loads(input_shape_list_str)
        input_shapes = [y for x in input_shape_list for y in x] # https://coderwall.com/p/rcmaea/flatten-a-list-of-lists-in-one-line-in-python
        input_width = input_shapes[0][0]
        input_height = input_shapes[0][1]
        input_channel = input_shapes[0][2]
    else:
        if args.input_height:
            input_height = args.input_height
        if args.input_width:
            input_width = args.input_width
        if args.input_channel:
            input_channel = args.input_channel

    if args.output_list_path:
        output_list_str = open(args.output_list_path, 'rb').read()
        outputs = json.loads(output_list_str)
        outputs = ['{0}/{0}'.format(output) for output in outputs]
    elif args.output:
        outputs = args.output

    if args.input_batch:
        input_batch = args.input_batch

    if args.model:
        model = args.model

    convert(model,inputs,outputs,(input_batch,input_height,input_width,input_channel))


if __name__ == '__main__':
    main()
