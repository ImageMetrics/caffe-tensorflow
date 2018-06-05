#!/usr/bin/env python

import json
import numpy as np
import argparse
from kaffe import KaffeError, print_stderr
from kaffe.tensorflow import TensorFlowTransformer
from kaffe.tensorflow.transformer import TensorFlowMapper

def fatal_error(msg):
    print_stderr(msg)
    exit(-1)


def validate_arguments(args):
    if (args.data_output_path is not None) and (args.caffemodel is None):
        fatal_error('No input data path provided.')
    if (args.caffemodel is not None) and (args.data_output_path is None):
        fatal_error('No output data path provided.')
    if (args.code_output_path is None) and (args.data_output_path is None):
        fatal_error('No output path specified.')


def convert(def_path, caffemodel_path, data_output_path, code_output_path, input_list_path, input_shape_list_path, output_list_path, phase):
    try:
        transformer = TensorFlowTransformer(def_path, caffemodel_path, phase=phase)
        print_stderr('Converting data...')
        if caffemodel_path is not None:
            data = transformer.transform_data()
            print_stderr('Saving data...')

            with open(data_output_path, 'wb') as data_out:
                np.save(data_out, data)
        if code_output_path:
            print_stderr('Saving source...')
            with open(code_output_path, 'wb') as src_out:
                source = transformer.transform_source()
                src_out.write(source)
        if input_list_path or input_shape_list_path or output_list_path:
            mapper = TensorFlowMapper(transformer.graph)
            chains = mapper.map()
            if input_list_path:
                inputs = [[chain[0].node.parents[0].name for node in chain[0].node.parents] for chain in chains]
                with open(input_list_path, 'wb') as input_list_out:
                    json.dump(inputs, input_list_out)
            if input_shape_list_path:
                shapes = [[[node.output_shape.width, node.output_shape.height, node.output_shape.channels] for node in chain[0].node.parents] for chain in chains]
                with open(input_shape_list_path, 'wb') as input_shape_list_out:
                    json.dump(shapes, input_shape_list_out)
            if output_list_path:
                outputs = [chain[-1].node.name for chain in chains]
                with open(output_list_path, 'wb') as output_list_out:
                    json.dump(outputs, output_list_out)
        print_stderr('Done.')
    except KaffeError as err:
        fatal_error('Error encountered: {}'.format(err))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('def_path', help='Model definition (.prototxt) path')
    parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
    parser.add_argument('--data-output-path', help='Converted data output path')
    parser.add_argument('--code-output-path', help='Save generated source to this path')
    parser.add_argument('--input-list-path', help='Input layer names list file path')
    parser.add_argument('--input-shape-list-path', help='Input layer shape list file path')
    parser.add_argument('--output-list-path', help='Output layer names list file path')
    parser.add_argument('-p',
                        '--phase',
                        default='test',
                        help='The phase to convert: test (default) or train')
    args = parser.parse_args()
    validate_arguments(args)
    convert(args.def_path, args.caffemodel, args.data_output_path, args.code_output_path,
            args.input_list_path, args.input_shape_list_path, args.output_list_path,
            args.phase)


if __name__ == '__main__':
    main()
