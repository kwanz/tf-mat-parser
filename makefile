CXXFILES = parse_mat.cc

TARGET = parse_mat.so

CXXFLAGS0 = -std=c++11 -shared

CXXFLAGS1 = -fPIC

LIBS = -lmatio -lz

TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

all:
	$(CXX) $(CXXFLAGS0) $(CXXFILES) $(CXXFLAGS1) -o $(TARGET) -I $(TF_INC) $(LIBS)
