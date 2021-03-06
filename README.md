# tf-mat-parser

## Introduction

A quick and dirty TensorFlow Reader Op for Matlab .MAT file parsing support.

## Requirements

* Requires the MatIO library (https://sourceforge.net/projects/matio/), and
* (optionally) its dependencies for Matlab 7.3 and compressed .MAT file support.

## Usage

1. Build the op with TensorFlow binary or source.
2. Load and run the op in Python

  ```
  import tensorflow as tf
  parse_mat_module = tf.load_op_library('parse_mat.so')
  mat_tensor = parse_mat_module.parse_mat([filename], [matrixname], dtype=[datatype])
  ```
3. To use the parser in the input pipeline (cf.
   https://www.tensorflow.org/how_tos/reading_data/index.html), use an
   `IdentityReader` to produce file name strings from a file name queue, and
   then feed the resulting tensor to `parse_mat`.

## Known Issues

* uint32 and uint64 types are not supported because of missing types in
   TensorFlow source as of r0.9: [#1894](https://github.com/tensorflow/tensorflow/issues/1894)

TODO:

* Add uint32 and uint64 support once TensorFlow supported types are updated
* Probably redo the op as a Reader to get rid of the `IdentityReader` thing
