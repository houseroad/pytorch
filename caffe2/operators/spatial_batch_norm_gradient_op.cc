#include "caffe2/operators/spatial_batch_norm_op.h"

#include <string>

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
template <typename T>
void SpatialBNGradientOp<CPUContext>::
    ComputeMultiBatchScaleBiasGradientsAndFusedParams(
        const int N,
        const int C,
        const int HxW,
        const T* scale,
        const T* mean,
        const T* rstd,
        const T* dscale_sum,
        const T* dbias_sum,
        T* dscale,
        T* dbias,
        T* alpha,
        T* beta,
        T* gamma) {
  ConstEigenVectorArrayMap<T> scale_arr(scale, C);
  ConstEigenVectorArrayMap<T> mean_arr(mean, C);
  ConstEigenVectorArrayMap<T> rstd_arr(rstd, C);
  EigenVectorArrayMap<T> dscale_arr(dscale, C);
  EigenVectorArrayMap<T> dbias_arr(dbias, C);
  EigenVectorArrayMap<T> alpha_arr(alpha, C);
  EigenVectorArrayMap<T> beta_arr(beta, C);
  EigenVectorArrayMap<T> gamma_arr(gamma, C);
  const T inv_num_batches = T(1) / static_cast<T>(num_batches_);
  math::Scale<T, T, CPUContext>(
      C, inv_num_batches, dscale_sum, dscale, &context_);
  math::Scale<T, T, CPUContext>(
      C, inv_num_batches, dbias_sum, dbias, &context_);
  const T inv_nhw = T(1) / static_cast<T>(N * HxW);
  alpha_arr = scale_arr * rstd_arr;
  beta_arr = dscale_arr * rstd_arr;
  gamma_arr = alpha_arr * (mean_arr * beta_arr - dbias_arr) * inv_nhw;
  beta_arr *= -alpha_arr * inv_nhw;
}

template <>
template <typename T>
void SpatialBNGradientOp<CPUContext>::ComputeScaleBiasGradientsAndFusedParams(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* scale,
    const T* mean,
    const T* rstd,
    T* dscale,
    T* dbias,
    T* alpha,
    T* beta,
    T* gamma,
    T* /* scratch */) {
  ConstEigenVectorArrayMap<T> scale_arr(scale, C);
  ConstEigenVectorArrayMap<T> mean_arr(mean, C);
  ConstEigenVectorArrayMap<T> rstd_arr(rstd, C);
  EigenVectorArrayMap<T> dscale_arr(dscale, C);
  EigenVectorArrayMap<T> dbias_arr(dbias, C);
  EigenVectorArrayMap<T> alpha_arr(alpha, C);
  EigenVectorArrayMap<T> beta_arr(beta, C);
  EigenVectorArrayMap<T> gamma_arr(gamma, C);
  if (order_ == StorageOrder::NCHW) {
    ConstEigenArrayMap<float> dY0_arr(dY, HxW, C);
    ConstEigenArrayMap<float> X0_arr(X, HxW, C);
    dscale_arr = (dY0_arr * X0_arr).colwise().sum();
    dbias_arr = dY0_arr.colwise().sum();
    for (int i = 1; i < N; ++i) {
      ConstEigenArrayMap<float> dYi_arr(dY + i * C * HxW, HxW, C);
      ConstEigenArrayMap<float> Xi_arr(X + i * C * HxW, HxW, C);
      dscale_arr += (dYi_arr * Xi_arr).colwise().sum();
      dbias_arr += dYi_arr.colwise().sum();
    }
  } else {
    ConstEigenArrayMap<float> dY_arr(dY, C, N * HxW);
    ConstEigenArrayMap<float> X_arr(X, C, N * HxW);
    dscale_arr = dY_arr.col(0) * X_arr.col(0);
    dbias_arr = dY_arr.col(0);
    for (int i = 1; i < N * HxW; ++i) {
      dscale_arr += dY_arr.col(i) * X_arr.col(i);
      dbias_arr += dY_arr.col(i);
    }
  }
  dscale_arr = (dscale_arr - mean_arr * dbias_arr) * rstd_arr;
  const T inv_nhw = T(1) / static_cast<T>(N * HxW);
  alpha_arr = scale_arr * rstd_arr;
  beta_arr = dscale_arr * rstd_arr;
  gamma_arr = alpha_arr * (mean_arr * beta_arr - dbias_arr) * inv_nhw;
  beta_arr *= -alpha_arr * inv_nhw;
}

template <>
template <typename T>
void SpatialBNGradientOp<CPUContext>::ComputeXGradient(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* alpha,
    const T* beta,
    const T* gamma,
    T* dX) {
  ConstEigenVectorArrayMap<T> alpha_arr(alpha, C);
  ConstEigenVectorArrayMap<T> beta_arr(beta, C);
  ConstEigenVectorArrayMap<T> gamma_arr(gamma, C);
  if (order_ == NCHW) {
    const int stride = C * HxW;
    const T* dY_ptr = dY;
    const T* X_ptr = X;
    T* dX_ptr = dX;
    for (int i = 0; i < N; ++i) {
      EigenArrayMap<T>(dX_ptr, HxW, C) =
          (ConstEigenArrayMap<T>(dY_ptr, HxW, C).rowwise() *
               alpha_arr.transpose() +
           ConstEigenArrayMap<T>(X_ptr, HxW, C).rowwise() *
               beta_arr.transpose())
              .rowwise() +
          gamma_arr.transpose();
      dY_ptr += stride;
      X_ptr += stride;
      dX_ptr += stride;
    }
  } else {
    EigenArrayMap<T>(dX, C, N * HxW) =
        (ConstEigenArrayMap<T>(dY, C, N * HxW).colwise() * alpha_arr +
         ConstEigenArrayMap<T>(X, C, N * HxW).colwise() * beta_arr)
            .colwise() +
        gamma_arr;
  }
}

REGISTER_CPU_OPERATOR(SpatialBNGradient, SpatialBNGradientOp<CPUContext>);

// Input: X, scale, dY, mean, variance, dscale, dbias
// Output: dX, dscale, dbias
OPERATOR_SCHEMA(SpatialBNGradient)
    .NumInputs({5, 7})
    .NumOutputs(3)
    .AllowInplace({{5, 1}, {6, 2}});

namespace {

// Spatial batch normalization's gradient, depending on the various input sizes,
// is a bit more complex than usual gradient operators.
class GetSpatialBNGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    // Check if we are in training or testing mode.
    const bool is_test =
        ArgumentHelper::GetSingleArgument(def_, OpSchema::Arg_IsTest, 0);
    const int num_batches =
        ArgumentHelper::GetSingleArgument(def_, "num_batches", 1);
    const std::vector<string> grad_outputs = {GI(0), GI(1), GI(2)};
    std::vector<string> grad_inputs;
    if (is_test) {
      // This is in testing mode. The operator should have five inputs:
      //     X, scale, bias, estimated_mean, estimated_variance
      // The gradient inputs are:
      //     X, scale, dY, estimated_mean, estimated_variance
      CAFFE_ENFORCE_EQ(def_.input_size(), 5);
      CAFFE_ENFORCE_EQ(def_.output_size(), 1);
      grad_inputs = std::vector<std::string>{I(0), I(1), GO(0), I(3), I(4)};
    } else if (num_batches > 1) {
      CAFFE_ENFORCE_EQ(def_.input_size(), 7);
      CAFFE_ENFORCE_EQ(def_.output_size(), 5);
      grad_inputs =
          std::vector<std::string>{I(0), I(1), GO(0), O(3), O(4), GI(1), GI(2)};
    } else {
      CAFFE_ENFORCE_EQ(def_.input_size(), 5);
      CAFFE_ENFORCE_EQ(def_.output_size(), 5);
      grad_inputs = std::vector<std::string>{I(0), I(1), GO(0), O(3), O(4)};
    }
    return SingleGradientDef(
        "SpatialBNGradient", "", grad_inputs, grad_outputs);
  }
};

} // namespace

REGISTER_GRADIENT(SpatialBN, GetSpatialBNGradient);

} // namespace caffe2
