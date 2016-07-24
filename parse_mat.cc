#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "../public/tensor_c_api.h"
#include <matio.h>

using namespace tensorflow;

REGISTER_OP("ParseMat")
    .Input("mat_path: string")
    .Input("var: string")
    .Output("output: out_type")
    .Attr("out_type: {float,double,uint8,int8,uint16,int16,int32,int64}");

template<typename T>
class ParseMatOp : public OpKernel {
public:
    explicit ParseMatOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("out_type", &out_type_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& mat_path_tensor = context->input(0);
        const Tensor& var_tensor = context->input(1);
        auto mat_path = mat_path_tensor.flat<string>();
        const char *mat_path_ch = mat_path(0).c_str();
        auto var = var_tensor.flat<string>();
        const char *var_ch = var(0).c_str();

        mat_t *matfp = NULL;
        matvar_t *matvar = NULL;
        matfp = Mat_Open(mat_path_ch, MAT_ACC_RDONLY);
        OP_REQUIRES(context, matfp, errors::NotFound("Failed to open mat file: ", mat_path_ch));
        matvar = Mat_VarRead(matfp, var_ch);
        OP_REQUIRES(context, matvar, errors::InvalidArgument("Matrix not found in mat file: ", var_ch));
        OP_REQUIRES(context,
                    matvar->class_type == MAT_C_DOUBLE ||
                    matvar->class_type == MAT_C_SINGLE ||
                    matvar->class_type == MAT_C_INT8 ||
                    matvar->class_type == MAT_C_UINT8 ||
                    matvar->class_type == MAT_C_INT16 ||
                    matvar->class_type == MAT_C_UINT16 ||
                    matvar->class_type == MAT_C_INT32 ||
                    matvar->class_type == MAT_C_UINT32 ||
                    matvar->class_type == MAT_C_INT64 ||
                    matvar->class_type == MAT_C_UINT64,
                    errors::InvalidArgument("Variable is not numerical type: ", matvar->class_type));
        OP_REQUIRES(context,
                    matvar->nbytes % sizeof(T) == 0,
                    errors::InvalidArgument("Matrix has ", matvar->nbytes, " bytes, ",
                                            " not a multiple of ", sizeof(T),
                                            ", the size of ", DataTypeString(out_type_)));

        Tensor* output_tensor = NULL;
        TensorShape output_shape = TensorShape();
        int32* output_dims = new int32[matvar->rank];
        for (int i = 0; i < matvar->rank; i++)
            output_dims[i] = (int32)matvar->dims[i];
        TensorShapeUtils::MakeShape(output_dims, (int64)matvar->rank, &output_shape);
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, output_shape, &output_tensor));
        auto output = output_tensor->flat<T>();
        for (int i = 0; i < output_shape.num_elements(); i++)
            output(i) = *((T*)matvar->data + i);
        // TODO: flip storage order

        Mat_VarFree(matvar);
        Mat_Close(matfp);
        delete[] output_dims;
        matvar = NULL;
        matfp = NULL;
    }

private:
    DataType out_type_;
};

#define REGISTER_PARSE_MAT(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("ParseMat").Device(DEVICE_CPU).TypeConstraint<type>("out_type"), \
      ParseMatOp<type>)

REGISTER_PARSE_MAT(float);
REGISTER_PARSE_MAT(double);
REGISTER_PARSE_MAT(uint8);
REGISTER_PARSE_MAT(int8);
REGISTER_PARSE_MAT(uint16);
REGISTER_PARSE_MAT(int16);
//REGISTER_PARSE_MAT(uint32);
REGISTER_PARSE_MAT(int32);
//REGISTER_PARSE_MAT(uint64);
REGISTER_PARSE_MAT(int64);

#undef REGISTER_PARSE_MAT
