// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void MySoftmaxLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "MySoftmax Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "MySoftmax Layer takes a single blob as output.";
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  softmax_bottom_vec_.clear();
  nClassLabel = this->layer_param_.my_softmax_param().stride();
  nRotation = bottom[0]->count() / (bottom[0]->num() * nClassLabel );
  // NOTICE bottom[0] should be reshaped!
  Blob<Dtype> bottomR( bottom[0]->num() * nRotation, nClassLabel, bottom[0]->height(), bottom[0]->width() );
  bottomR.set_cpu_data( (Dtype*)bottom[0]->cpu_data() );
  softmax_bottom_vec_.push_back(&bottomR);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
Dtype MySoftmaxLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  // NOTICE bottom[0] should be reshaped!
  Blob<Dtype> bottomR( bottom[0]->num() * nRotation, nClassLabel, bottom[0]->height(), bottom[0]->width() );
  bottomR.set_cpu_data( (Dtype*)bottom[0]->cpu_data() );
  softmax_bottom_vec_[0] = &bottomR;
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();

  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  memcpy(top_data, prob_data, sizeof(Dtype) * bottom[0]->count());

  return Dtype(0);
}

template <typename Dtype>
void MySoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // NOT IMPLEMENTED
}


INSTANTIATE_CLASS(MySoftmaxLayer);


}  // namespace caffe
