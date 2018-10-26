#include <cstdio>
#include <string>

#include <gtest/gtest.h>

#include "caffe2/serialize/intermediate_model.h"

namespace at {
namespace {

TEST(IntermediateModel, SerializeAndDeserialize) {

  // initialize the imodel for test
  std::string model_name("Test-Model-Name");
  std::string producer_name("Test-Producer-Name");
  std::string producer_version("Test-Producer-Version");
  int64_t proto_version = 1;

  at::serialize::IntermediateModel imodel;
  imodel.setName(model_name);
  imodel.setProducerName(producer_name);
  imodel.setProducerVersion(producer_version);
  imodel.setProtoVersion(proto_version);

  at::serialize::IntermediateModule* main_module = imodel.mutableMainModule();
  std::string module_name("Test-Module-Name");
  main_module->setName(module_name);
  std::vector<serialize::IntermediateModule>* subs = main_module->mutableSubmodules();
  subs->resize(1);
  at::serialize::IntermediateModule& sub_module = subs->at(0);
  std::string sub_name("Test-Submodule-Name");
  sub_module.setName(sub_name);

  std::vector<serialize::IntermediateParameter>* params = main_module->mutableParameters();
  params->resize(1);
  at::serialize::IntermediateParameter& param = params->at(0);
  std::string param_name("Test-Parameter-Name");
  param.setName(param_name);
  bool is_buffer = true;
  param.setIsBuffer(is_buffer);
  bool require_gradient = true;
  param.setRequireGradient(require_gradient);
  at::serialize::IntermediateTensor* tensor = param.mutableTensor();
  std::vector<int64_t>* dims = tensor->mutableDims();
  size_t raw_size = sizeof(float);
  for (size_t i = 2; i < 5; ++i) {
    raw_size *= i;
    dims->push_back(i);
  }
  tensor->setDataType(caffe2::TensorProto_DataType_FLOAT);
  std::vector<char> data_vector;
  data_vector.resize(raw_size);
  for (size_t i = 0; i < data_vector.size(); ++i) {
    data_vector[i] = data_vector.size() - i;
  }
  at::DataPtr data_ptr(data_vector.data(), at::kCPU);
  std::shared_ptr<at::serialize::SharedData> data = std::make_shared<at::serialize::SharedData>(0, std::move(data_ptr));
  tensor->setData(data);
  auto* device_detail = tensor->mutableDeviceDetail();

  std::string tmp_name = std::tmpnam(nullptr);
  at::serialize::serializeIntermediateModel(&imodel, tmp_name);

  at::serialize::IntermediateModel loaded_model;
  at::serialize::deserializeIntermediateModel(&loaded_model, tmp_name);

  ASSERT_EQ(loaded_model.name(), model_name);
}

}  // namespace
}  // namespace at
