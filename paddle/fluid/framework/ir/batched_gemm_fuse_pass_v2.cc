/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/framework/ir/batched_gemm_fuse_pass_v2.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include <string>

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle



#define MAX_COLUMNS 80

namespace paddle {
namespace framework {
namespace ir {

void BuildBatchedGemmFusePattern_v2(PDPattern* pattern,
                                  const std::string& name_scope,
                                  int num_columns, int num_layers) {
  
  VLOG(4) << "in BuildBatchedGemmFusePattern_v2";
  auto is_split_op_with_outputs = [](Node* x, int num) -> bool {
    return x && x->IsOp() && x->Op()->Type() == "split" &&
           x->Op()->Output("Out").size() == static_cast<size_t>(num);
  };

  auto is_nth_output_var_of_split = [=](Node* x, int idx) -> bool {
    return x && x->IsVar() && VarLinksFromOp(x, "split") &&
           x->outputs.size() == 1 && IsNthOutput(x, x->inputs[0], "Out", idx);
  };
  
  //layer 1
  auto is_nth_output_fc_op_of_split = [=](Node* x, int idx) -> bool {
    return x && x->IsOp() && x->Op()->Type() == "fc" && 
           is_nth_output_var_of_split(x->inputs[0], idx);
  };
  
  auto is_output_var_of_fc = [=](Node* x) -> bool {
    return x && x->IsVar() && VarLinksFromOp(x, "fc");
  };
  auto is_input_var_of_fc = [=](Node* x) -> bool {
    return x && x->IsVar() && VarLinksToOp(x, "fc");
  };

  auto is_output_var_of_elementwise_add = [=](Node* x) -> bool {
    return x && x->IsVar() && VarLinksFromOp(x, "elementwise_add");
  };
  /*
  auto is_input_var_of_elementwise_add = [=](Node* x) -> bool {
    return x && x->IsVar() && VarLinksToOp(x, "elementwise_add");
  };
  */

  auto is_input_var_of_split = [=](Node* x) -> bool {
    return x && x->IsVar() && VarLinksToOp(x, "split");
  };

  auto is_param_fc = [=](Node* n, const std::string& param_name)->bool {
    for (auto* out : n->outputs) {
      if (out->IsOp() && out->Op()->Type() == "fc" &&
          n->Name() == out->Op()->Input(param_name)[0]) {
        return true;
      }
    }
    return false;
  };

  //param_name: X or Y
  auto is_param_elementwise_add = [=](Node* n, const std::string& param_name)->bool {
    for (auto* out : n->outputs) {
      if (out->IsOp() && out->Op()->Type() == "elementwise_add" &&
          n->Name() == out->Op()->Input(param_name)[0]) {
        return true;
      }
    }
    return false;
  };

  
  //第idx个输出，第n层   n= 0,1,...num_layer-1
  //第n层fc的第idx个输入
  auto is_mth_fc_op_of_layer_n = [=](Node* x, int idx, int n) -> bool {
      bool cond = true;
      Node * tmp = x;
      for(int i = 0; i < n; i++)//往上追溯n层
      {
          bool fine = tmp && tmp->IsOp() && tmp->Op()->Type() == "fc"  && 
                      is_output_var_of_fc(tmp->inputs[0]) && is_input_var_of_fc(tmp->inputs[0]) &&
                      tmp->inputs[0]->inputs[0]->IsOp() && tmp->inputs[0]->inputs[0]->Op()->Type() == "fc";
          if(fine != true)
          {
            cond = false;
            break;
          }
          tmp = tmp->inputs[0]->inputs[0];//last layer of fc
      }
      //tmp <-->layer1
      if(cond)
          return is_nth_output_fc_op_of_split(tmp, idx);
      return cond;
  };
  
  //input var
  PDNode* split_input_var = pattern->NewNode(
      [=](Node* x) { 
        bool basic = x && x->IsVar() && is_input_var_of_split(x);
        return basic;
       },
      name_scope + "/split_input")->assert_is_op_input("split", "X")->AsInput();

  PDNode* split_op = pattern->NewNode(
      [=](Node* x) { return is_split_op_with_outputs(x, num_columns); },
      name_scope + "/split_op");

  split_op->LinksFrom({split_input_var});
  //layer 1
  std::vector<PDNode*> fc_ops_input_var_x(num_columns);
  std::vector<PDNode*> fc_ops_input_var_w(num_columns);
  std::vector<PDNode*> fc_ops_input_var_bias(num_columns);
  std::vector<PDNode*> fc_ops_output_var(num_columns);
  std::vector<PDNode*> fc_ops(num_columns);
  
  for (int i = 0; i < num_columns; ++i) {
    //layer 1 fc_ops
    fc_ops[i] = pattern->NewNode(
        [=](Node* x) {
          return x && x->IsOp() && is_nth_output_fc_op_of_split(x, i);
        },
        name_scope + "/fc_op_" + std::to_string(i) + "_layer_1");
    
    fc_ops_input_var_x[i] = pattern->NewNode(
        [=](Node* x) {
          bool basic = x && x->IsVar() && is_input_var_of_fc(x) &&
                       is_nth_output_var_of_split(x, i);
          return basic;
        },
        name_scope + "/fc_in_input_" + std::to_string(i) + "_layer_1")->assert_is_op_input("fc", "Input")->AsIntermediate();

    
    fc_ops_input_var_bias[i] = pattern->NewNode(
        [=](Node* x) {
          bool basic = x && x->IsVar() && x->outputs.size()==1 && x->inputs.empty() &&
          is_nth_output_fc_op_of_split(x->outputs[0], i);
          
          /*for (auto* out : x->outputs) {
            if (out->Op()->Type() == "fc" &&
                x->Name() == out->Op()->Input("Bias")[0]) {
              return basic;
            }
          }*/
          return basic && is_param_fc(x,"Bias");
        },
        name_scope + "/fc_in_bias_" + std::to_string(i) + "_layer_1")->assert_is_op_input("fc", "Bias")->AsIntermediate();

    fc_ops_input_var_w[i] = pattern->NewNode(
        [=](Node* x) {
          bool basic = x && x->IsVar() && x->outputs.size()==1 && x->inputs.empty() &&
                        is_nth_output_fc_op_of_split(x->outputs[0], i);
          /*for (auto* out : x->outputs) {
            if (out->Op()->Type() == "fc" &&
                x->Name() == out->Op()->Input("W")[0]) {
              return basic;
            }
          }*/
          return basic && is_param_fc(x,"W");
        },
        name_scope + "/fc_in_w_" + std::to_string(i) + "_layer_1")->assert_is_op_input("fc", "W")->AsIntermediate();

    fc_ops_output_var[i] = pattern->NewNode(
        [=](Node* x) {
          bool basic = x && x->IsVar() && is_output_var_of_fc(x) &&
                       is_nth_output_fc_op_of_split(x->inputs[0], i) && is_param_elementwise_add(x, "X"); //add last assertion to fit statlas model struct
          return basic;
        },
        //name_scope + "/fc_output_" + std::to_string(i) + "_layer_1")->assert_is_op_output("fc", "Out")->AsOutput();
        name_scope + "/fc_output_" + std::to_string(i) + "_layer_1")->assert_is_op_output("fc", "Out");
    fc_ops_output_var[i] = fc_ops_output_var[i]->AsIntermediate();
    // Links
    split_op->LinksTo({fc_ops_input_var_x[i]});
    fc_ops[i]
      ->LinksFrom({fc_ops_input_var_x[i], fc_ops_input_var_bias[i], fc_ops_input_var_w[i]})
      .LinksTo({fc_ops_output_var[i]});
  }

  //resnet part
  std::vector<PDNode*> layers_res_fc_ops_input_var_w(num_columns * 2);
  std::vector<PDNode*> layers_res_fc_ops_input_var_bias(num_columns * 2);
  std::vector<PDNode*> layers_res_fc_ops_output_var(num_columns * 2);
  std::vector<PDNode*> layers_res_fc_ops(num_columns * 2);

  std::vector<PDNode*> layers_elelmentwise_add_ops(num_columns);
  //std::vector<PDNode*> layers_elelmentwise_add_var_x(num_columns); //shortcut
  //std::vector<PDNode*> layers_elelmentwise_add_var_y(num_columns);
  std::vector<PDNode*> layers_elelmentwise_add_output_var(num_columns);

  for(int j = 0; j < num_columns; j++)
  {
    //layer 2 fc_ops  && layer 1 in resnet struct
    layers_res_fc_ops[j] = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsOp() && is_mth_fc_op_of_layer_n(x, j, 1);
      },
      name_scope + "/fc_op_" + std::to_string(j) + "_layer_2");

    layers_res_fc_ops_input_var_w[j] = pattern->NewNode(
      [=](Node* x) {
        bool basic = x && x->IsVar() && x->outputs.size()==1 && x->inputs.empty() && is_mth_fc_op_of_layer_n(x->outputs[0], j, 1);
        return basic && is_param_fc(x,"W");
      },
      name_scope + "/fc_in_w_" + std::to_string(j) + "_layer_2")->assert_is_op_input("fc", "W")->AsIntermediate();

    layers_res_fc_ops_input_var_bias[j] = pattern->NewNode(
      [=](Node* x) {
        bool basic = x && x->IsVar() && x->outputs.size()==1 && x->inputs.empty() && is_mth_fc_op_of_layer_n(x->outputs[0], j, 1);
        return basic && is_param_fc(x,"Bias");
      },
      name_scope + "/fc_in_bias_" + std::to_string(j) + "_layer_2")->assert_is_op_input("fc", "Bias")->AsIntermediate();
    
    layers_res_fc_ops_output_var[j] = pattern->NewNode(
      [=](Node* x) {
        bool basic = x && x->IsVar() && is_output_var_of_fc(x) && x->outputs.size() >= 1 &&
                      is_mth_fc_op_of_layer_n(x->inputs[0], j, 1);
        return basic;
      },
      name_scope + "/fc_output_" + std::to_string(j) + "_layer_2")->assert_is_op_output("fc", "Out")->AsIntermediate();;


    //layer 3 fc_ops  && layer 2 in resnet struct
    layers_res_fc_ops[j + num_columns] = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsOp() && is_mth_fc_op_of_layer_n(x, j, 2);
      },
      name_scope + "/fc_op_" + std::to_string(j) + "_layer_3");

    layers_res_fc_ops_input_var_w[j + num_columns] = pattern->NewNode(
      [=](Node* x) {
        bool basic = x && x->IsVar() && x->outputs.size()==1 && x->inputs.empty() && is_mth_fc_op_of_layer_n(x->outputs[0], j, 2);
        return basic && is_param_fc(x,"W");
      },
      name_scope + "/fc_in_w_" + std::to_string(j) + "_layer_3")->assert_is_op_input("fc", "W")->AsIntermediate();

    layers_res_fc_ops_input_var_bias[j + num_columns] = pattern->NewNode(
      [=](Node* x) {
        bool basic = x && x->IsVar() && x->outputs.size()==1 && x->inputs.empty() && is_mth_fc_op_of_layer_n(x->outputs[0], j, 2);
        return basic && is_param_fc(x, "Bias");
      },
      name_scope + "/fc_in_bias_" + std::to_string(j) + "_layer_3")->assert_is_op_input("fc", "Bias")->AsIntermediate();
    
    layers_res_fc_ops_output_var[j + num_columns] = pattern->NewNode(
      [=](Node* x) {
        bool basic = x && x->IsVar() && is_output_var_of_fc(x) && x->outputs.size() >= 1 &&
                      is_mth_fc_op_of_layer_n(x->inputs[0], j, 2) && is_param_elementwise_add(x, "Y"); //add last assertion to fit statlas model struct
        return basic;
      },
      name_scope + "/fc_output_" + std::to_string(j) + "_layer_3")->assert_is_op_output("fc", "Out")->AsIntermediate();

    //condition is not too strict, may cause more time to search
    layers_elelmentwise_add_ops[j] = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "elementwise_add" && x->inputs.size()>=1; //&& is_mth_fc_op_of_layer_n(x->inputs[0]->inputs[0], j, 1) //shortcut 
                //&& is_mth_fc_op_of_layer_n(x->inputs[1]->inputs[0], j, 2);//nearlest
      },
      name_scope + "/elementwise_op_" + std::to_string(j));
    
    layers_elelmentwise_add_output_var[j] = pattern->NewNode(
      [=](Node* x) {
        bool basic = x && x->IsVar() && is_output_var_of_elementwise_add(x) && x->inputs.size() >= 1; //&&
                    //is_mth_fc_op_of_layer_n(x->inputs[0]->inputs[0]->inputs[0], j, 1) &&//shortcut 
                    //is_mth_fc_op_of_layer_n(x->inputs[0]->inputs[1]->inputs[0], j, 2);//nearlest
        return basic;
      },
      name_scope + "/elementwise_output_" + std::to_string(j))->assert_is_op_output("elementwise_add", "Out")->AsOutput();
    
    //link layer2 
    layers_res_fc_ops[j]
      ->LinksFrom({fc_ops_output_var[j], layers_res_fc_ops_input_var_bias[j], layers_res_fc_ops_input_var_w[j]})
      .LinksTo({layers_res_fc_ops_output_var[j]});

    //link layer3
    layers_res_fc_ops[j + num_columns]
      ->LinksFrom({layers_res_fc_ops_output_var[j], layers_res_fc_ops_input_var_bias[j + num_columns], layers_res_fc_ops_input_var_w[j + num_columns]})
      .LinksTo({layers_res_fc_ops_output_var[j + num_columns]});

    //link layer elementwise_add
    layers_elelmentwise_add_ops[j]
    ->LinksFrom({fc_ops_output_var[j], layers_res_fc_ops_output_var[j + num_columns]})
      .LinksTo({layers_elelmentwise_add_output_var[j]});
  }
  
  return;
}

static int BuildFusion_v2(Graph* graph, const std::string& name_scope,
                       int num_columns) {
  int num_layers = 3; //3 of fc layer and 1 elementwise_add layer
  //get scpoe
  PADDLE_ENFORCE_EQ(graph->Has(kParamScopeAttr), true,
                    platform::errors::InvalidArgument(
                        "Graph must have kParamScopeAttr attribute."));
  //auto* scope = graph->Get<framework::Scope>(kParamScopeAttr);
  auto& scope = graph->Get<Scope>(kParamScopeAttr);
  
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  VLOG(3) << "pattern string " + pattern->DotString();
  BuildBatchedGemmFusePattern_v2(pattern, name_scope, num_columns, num_layers);

  auto retrieve_node = [](const std::string& name,
                          const GraphPatternDetector::subgraph_t& subgraph,
                          const PDPattern& pat) -> Node* {
    PADDLE_ENFORCE_GT(subgraph.count(pat.RetrieveNode(name)), 0,
                      platform::errors::NotFound(
                          "Pattern has no node called %s.", name.c_str()));
    Node* p = subgraph.at(pat.RetrieveNode(name));
    PADDLE_ENFORCE_NOT_NULL(p, platform::errors::NotFound(
                                   "Subgraph has no node %s.", name.c_str()));
    return p;
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle BatchedGemm fuse";
    std::vector<std::string> input_names_input(num_columns);
    std::vector<std::string> input_names_w(num_columns * num_layers);
    std::vector<std::string> input_names_bias(num_columns * num_layers);
    std::vector<std::string> orig_ops_names(num_columns * num_layers);
    std::vector<std::string> output_names(num_columns * num_layers);
    std::vector<Node*> input_vars_input(num_columns);
    std::vector<Node*> input_vars_w(num_columns * num_layers);
    std::vector<Node*> input_vars_bias(num_columns * num_layers);
    std::vector<Node*> orig_ops(num_columns * num_layers);
    std::vector<Node*> output_vars(num_columns * num_layers);

    //newly add var
    std::vector<Node*> fc_output_vars(num_layers);
    std::vector<std::string> fc_output_names(num_layers);

    std::vector<Node*> elementwise_add_output_vars(1);
    std::vector<std::string> elementwise_add_output_names(1);

    //elementwise_add about
    std::vector<std::string> elementwise_add_ops_names(num_columns);
    std::vector<Node*> elementwise_add_ops(num_columns);

    std::vector<std::string> output_elementwise_add_vars_names(num_columns);
    std::vector<Node*> output_elementwise_add_vars(num_columns);

    auto& fused_pattern = gpd.pattern();

    Node* node_split_input_var =  retrieve_node(name_scope + "/split_input", subgraph, fused_pattern);
    std::string node_split_input_var_name = node_split_input_var->Name();
    Node* node_split_op =  retrieve_node(name_scope + "/split_op", subgraph, fused_pattern);

    for(int i = 0; i < num_layers; i++){
      for(int j = 0; j < num_columns; j++){

        orig_ops[i* num_columns + j] =
          retrieve_node(name_scope + "/fc_op_" + std::to_string(j) + "_layer_" + std::to_string(i+1),
                        subgraph, fused_pattern);
        orig_ops_names[i* num_columns + j] = orig_ops[i* num_columns + j]->Name();
        if (i == 0)
        {
          input_vars_input[i* num_columns + j] =
            retrieve_node(name_scope + "/fc_in_input_" + std::to_string(j) + "_layer_" + std::to_string(i+1),
                          subgraph, fused_pattern);
          input_names_input[i* num_columns + j] = input_vars_input[i* num_columns + j]->Name();
        }

        input_vars_w[i* num_columns + j] =
          retrieve_node(name_scope + "/fc_in_w_" + std::to_string(j) + "_layer_" + std::to_string(i+1),
                        subgraph, fused_pattern);
        input_names_w[i* num_columns + j] = input_vars_w[i* num_columns + j]->Name();

        input_vars_bias[i* num_columns + j] =
          retrieve_node(name_scope + "/fc_in_bias_" + std::to_string(j) + "_layer_" +  std::to_string(i+1),
                        subgraph, fused_pattern);
        input_names_bias[i* num_columns + j] = input_vars_bias[i* num_columns + j]->Name();

        output_vars[i* num_columns + j] =
          retrieve_node(name_scope + "/fc_output_" + std::to_string(j) + "_layer_" + std::to_string(i+1),
                        subgraph, fused_pattern);
        output_names[i* num_columns + j] = output_vars[i* num_columns + j]->Name();

      }
    }
    for(int j = 0; j < num_columns; j++){
      elementwise_add_ops[j] =
        retrieve_node(name_scope + "/elementwise_op_" + std::to_string(j),
                      subgraph, fused_pattern);
      elementwise_add_ops_names[j] = elementwise_add_ops[j]->Name();

      output_elementwise_add_vars[j] =
        retrieve_node(name_scope + "/elementwise_output_" + std::to_string(j),
                      subgraph, fused_pattern);
      output_elementwise_add_vars_names[j] = output_elementwise_add_vars[j]->Name();
    }


    VLOG(4) << "my_special_flag";
    VLOG(4) << node_split_input_var_name;  
    for(auto item:input_names_input)
       VLOG(4) << item; 
    for(auto item:orig_ops_names)
       VLOG(4) << item;        
    for(auto item:output_names)
       VLOG(4) << item; 
    for(auto item:input_names_w)
       VLOG(4) << item; 
    for(auto item:input_names_bias)
       VLOG(4) << item;
    for(auto item:elementwise_add_ops_names)
       VLOG(4) << item;
    for(auto item:output_elementwise_add_vars_names)
       VLOG(4) << item;

    for(int i = 0; i < num_layers; i++){
      VLOG(4) << "****layer "<<i <<" ****";
      auto* fc_op0 = retrieve_node(name_scope + "/fc_op_0_layer_" + std::to_string(i+1),
                                        subgraph, fused_pattern);
      //combine w and bias
      //reuse first w and bias every layer
      auto* w0_tensor = scope.FindVar(input_vars_w[i* num_columns]->Name())->GetMutable<LoDTensor>();
      //auto* w0_data = w0_tensor->mutable_data<float>(platform::CPUPlace());
      
      auto* b0_tensor = scope.FindVar(input_vars_bias[i* num_columns]->Name())->GetMutable<LoDTensor>();
      //auto* b0_data = b0_tensor->mutable_data<float>(platform::CPUPlace());

      // Only support 2D-Tensor as weight for FC
      std::vector<int64_t> w_shape = input_vars_w[i* num_columns]->Var()->GetShape();
      size_t w_rank = w_shape.size();
      VLOG(4) << "w_rank: "<<w_rank;
      for(auto item:w_shape)
          VLOG(4)<<item;
      if (w_rank != 2) {
        VLOG(4)<<"w_ran != 2";
        for(auto item:w_shape)
          VLOG(4)<<item;
        return;
      }
      
      // Shape of bias should be [1, out_size] or [out_size]
      std::vector<int64_t> b_shape = input_vars_bias[i* num_columns]->Var()->GetShape();
      size_t b_rank = b_shape.size();
      VLOG(4) <<"b_rank: "<<b_rank;
      for(auto item:b_shape)
          VLOG(4)<<item;
      if (b_shape.size() == 1) {
        if (b_shape[0] != w_shape[1]) {
          return;
        }
      } else if (b_shape.size() == 2) {
        if (b_shape[0] != 1 || b_shape[1] != w_shape[1]) {
          return;
        }
      } else {
        return;
      }

      //提前申请显存空间
      //w拼接  batchcount * k * n
      //reserve first tensor value
      framework::LoDTensor tmp_w0_tensor;
      tmp_w0_tensor.Resize({w_shape[0], w_shape[1]});
      auto* tmp_w0_data = tmp_w0_tensor.mutable_data<float>(platform::CPUPlace());
      auto* w0_data = w0_tensor->mutable_data<float>(platform::CPUPlace());
      for(int k = 0; k < w_shape[0]* w_shape[1]; k++)
      {
        tmp_w0_data[k] = w0_data[k];
      }

      auto combined_w_dims = phi::make_ddim({num_columns, w_shape[0], w_shape[1]});
      auto* combined_w_desc = input_vars_w[i* num_columns]->Var();
      combined_w_desc->SetShape({num_columns, w_shape[0], w_shape[1]});
      combined_w_desc->SetPersistable(true);
      
      framework::LoDTensor tmp_combined_w_tensor;
      tmp_combined_w_tensor.Resize(combined_w_dims);
      w0_tensor->Resize(combined_w_dims);
      w0_data = w0_tensor->mutable_data<float>(platform::CPUPlace());//after resize
      auto* tmp_combined_w_data =
          tmp_combined_w_tensor.mutable_data<float>(platform::CPUPlace());
      
      int block_size = w_shape[0]* w_shape[1];
      //store first tensor to combined tensor
      for(int k = 0; k < w_shape[0]* w_shape[1]; k++)
      {
        tmp_combined_w_data[k] = tmp_w0_data[k];
      }
      //stro other(for 1 to column-1) tensor to combined tensor
      for(int j = 1; j < num_columns; j++)
      {
        auto* w_tensor = scope.FindVar(input_vars_w[i* num_columns + j]->Name())->GetMutable<LoDTensor>();
        auto* w_data = w_tensor->mutable_data<float>(platform::CPUPlace());
        int offset = j * block_size;
        for(int k = 0; k < block_size; k++)
            tmp_combined_w_data[offset + k] = w_data[k];
      }
      
      memcpy(w0_data, tmp_combined_w_data,
            sizeof(float) * w0_tensor->numel());

      VLOG(4) <<"combined_w ok";
      /*
      VLOG(3) <<"***check w in pass_v2*****";
      for(int j = 0; j < num_columns; j++)
      {
        std::string s_orig;
        std::string s_combine;
        auto* w_tensor = scope.FindVar(input_vars_w[i* num_columns + j]->Name())->GetMutable<LoDTensor>();
        auto* w_data = w_tensor->mutable_data<float>(platform::CPUPlace());
        int offset = j * block_size;
        for(int k = 0; k < block_size; k++)
        {
          s_orig += std::to_string(w_data[k]) + " ";
          s_combine += std::to_string(w0_data[offset + k]) + " ";
        }
        VLOG(3) <<"orig bc:"<<std::to_string(j)<<" "<<s_orig;
        VLOG(3) <<"combine bc:"<<std::to_string(j)<<" "<<s_combine;
      }
      */

      //2维b按照列拼接，1维直接罗列
      framework::LoDTensor tmp_b0_tensor;
      tmp_b0_tensor.Resize({b_shape[0]});
      auto* tmp_b0_data = tmp_b0_tensor.mutable_data<float>(platform::CPUPlace());
      auto* b0_data = b0_tensor->mutable_data<float>(platform::CPUPlace());
      for(int k = 0; k < b_shape[0]; k++)
      {
        tmp_b0_data[k] = b0_data[k];
      }

      auto* combined_b_desc = input_vars_bias[i* num_columns]->Var();
      auto combined_bias_dims = phi::make_ddim({num_columns, b_shape[0]});
      if (b_rank == 1)
      {
        combined_b_desc->SetShape({num_columns, b_shape[0]});
        block_size = b_shape[0];
      }
      else
      {
        VLOG(4) <<"not supported 2d bias";
        return;
        //combined_bias_dims = phi::make_ddim({b_shape[0]* num_columns, b_shape[1] * num_columns});
        //combined_b_desc->SetShape({b_shape[0]* num_columns, b_shape[1] * num_columns});
        //block_size = b_shape[0] * b_shape[1];
      }
      combined_b_desc->SetPersistable(true);
      framework::LoDTensor tmp_combined_b_tensor;
      tmp_combined_b_tensor.Resize(combined_bias_dims);
      VLOG(4) <<"resize tmp tensor ok";
      b0_tensor->Resize(combined_bias_dims);
      VLOG(4) <<"resize b0_tensor ok";
      b0_data = b0_tensor->mutable_data<float>(platform::CPUPlace());
      auto* tmp_combined_b_data =
          tmp_combined_b_tensor.mutable_data<float>(platform::CPUPlace());
      
      //store first tensor to tmp_combined_b_data
      for(int k = 0; k < b_shape[0]; k++)
      {
        tmp_combined_b_data[k] = tmp_b0_data[k];
      }
      //store other tensors to tmp_combined_b_data
      for(int j = 1; j < num_columns; j++)
      {
        auto* b_tensor = scope.FindVar(input_vars_bias[i* num_columns + j]->Name())->GetMutable<LoDTensor>();
        auto* b_data = b_tensor->mutable_data<float>(platform::CPUPlace());
        int offset = j * block_size;
        for(int k = 0; k < block_size; k++)
            tmp_combined_b_data[offset + k] = b_data[k];
      }
      VLOG(4) <<"assign tmp value ok";
      memcpy(b0_data, tmp_combined_b_data,
            sizeof(float) * b0_tensor->numel());
      VLOG(4) <<"combined_b ok";
      /*
      VLOG(3) <<"***check bias in pass_v2*****";
      for(int j = 0; j < num_columns; j++)
      {
        std::string s_orig;
        std::string s_combine;
        int offset = j * block_size;
        auto* b_tensor = scope.FindVar(input_vars_bias[i* num_columns + j]->Name())->GetMutable<LoDTensor>();
        auto* b_data = b_tensor->mutable_data<float>(platform::CPUPlace());

        for(int k = 0; k < block_size; k++)
        {
          s_orig += std::to_string(b_data[k]) + " ";
          s_combine += std::to_string(b0_data[offset + k]) + " ";
        }
        VLOG(3) <<"orig bc:"<<std::to_string(j)<<" "<<s_orig;
        VLOG(3) <<"combine bc:"<<std::to_string(j)<<" "<<s_combine;
      }
      */

      //w和bias应该是合并后的
      input_vars_bias[i* num_columns]->outputs.clear();
      input_vars_w[i* num_columns]->outputs.clear();
      if(i == 0)
      {
        node_split_input_var->outputs.clear();
        node_split_op->inputs.clear();
        node_split_op->outputs.clear();
      }

      const std::string& tmp_out_name_v2 = patterns::UniqueKey("new_batchedgemm_out_tmp_v2_"+ std::to_string(i));
      fc_output_names[i] = tmp_out_name_v2;
      auto* fc_out_var_desc =
        orig_ops[0]->Op()->Block()->Var(fc_output_names[i]);
      fc_out_var_desc->SetPersistable(false);
      std::vector<int64_t> tmp_input_shape = input_vars_input[0]->Var()->GetShape();
      fc_out_var_desc->SetShape({num_columns, tmp_input_shape[0], w_shape[1]}); //bc,batch,n
      fc_out_var_desc->SetDataType(input_vars_input[0]->Var()->GetDataType());
      auto* fc_out_var = g->CreateVarNode(fc_out_var_desc);
      fc_output_vars[i] = fc_out_var;
      
      //VarDesc fc_out_var_desc("new_batchedgemm_out_tmp_v2_"+ std::to_string(i)); //avoid exist
      //fc_out_var_desc.SetPersistable(false);
      //construct shape
      //std::vector<int64_t> tmp_input_shape = input_vars_input[0]->Var()->GetShape();
      //fc_out_var_desc.SetShape({num_columns, tmp_input_shape[0], w_shape[1]}); //bc,batch,n
      //auto* fc_out_var = g->CreateVarNode(&fc_out_var_desc);

      //fc_output_vars[i] = fc_out_var;
      //fc_output_names[i] = "new_batchedgemm_out_tmp_v2_"+ std::to_string(i);

      VLOG(4) <<"create fc out var ok";
      // Create New OpDesc
      OpDesc op_desc;
      op_desc.SetType("batchedgemm");
      VLOG(4) <<"set type ok";
      if(i == 0)
        op_desc.SetInput("Input", {node_split_input_var_name});
      else
        op_desc.SetInput("Input", {fc_output_names[i-1]});
      VLOG(4) <<"set input ok";
      op_desc.SetInput("W", {input_names_w[i* num_columns]});
      VLOG(4) <<"set W ok";
      op_desc.SetInput("Bias", {input_names_bias[i* num_columns]});
      VLOG(4) <<"set bias ok";
      //op_desc.SetOutput("Out", {"fc_out_tmp_"+ std::to_string(i)});
      op_desc.SetOutput("Out", {fc_output_names[i]});
      VLOG(4) <<"set output ok";
      //确保每一个fc的in_num_col_dims， activation_type都是一致的
      op_desc.SetAttr("in_num_col_dims", fc_op0->Op()->GetAttr("in_num_col_dims"));
      VLOG(4) <<"set in_num_col_dims ok";
      //std::string activation_type = with_relu ? "relu" : "";
      op_desc.SetAttr("activation_type", fc_op0->Op()->GetAttr("activation_type"));
      VLOG(4) <<"set attr ok";
      op_desc.Flush();
      VLOG(4) <<"set flush ok";
      auto* new_fc = graph->CreateOpNode(&op_desc);
      VLOG(4) <<"create fc ok";
      
      IR_NODE_LINK_TO(input_vars_bias[i* num_columns], new_fc);
      IR_NODE_LINK_TO(input_vars_w[i* num_columns], new_fc);
      IR_NODE_LINK_TO(new_fc, fc_out_var);

      if (i == 0)
      {
        IR_NODE_LINK_TO(node_split_input_var, new_fc); //复用第一个输入
      }
      else
      {
        IR_NODE_LINK_TO(fc_output_vars[i-1], new_fc);
      }
      
    } //end of i (num_layers)

    const std::string& tmp_add_name_v2 = patterns::UniqueKey("new_elementwise_add_out_tmp");
    VarDesc elementwise_add_out_var_desc(tmp_add_name_v2); //avoid exist
    elementwise_add_out_var_desc.SetPersistable(false);
    
    auto* elementwise_out_var = g->CreateVarNode(&elementwise_add_out_var_desc);
    elementwise_add_output_vars[0] = elementwise_out_var;
    elementwise_add_output_names[0] = tmp_add_name_v2;

    // Create elementwise_add
    OpDesc elementwise_add_op_desc(orig_ops[0]->Op()->Block());
    elementwise_add_op_desc.SetType("elementwise_add");
    VLOG(4) <<"set elementwise_add op ok";
    elementwise_add_op_desc.SetAttr("axis", 0);
    elementwise_add_op_desc.SetInput("X", {fc_output_names[0]});
    elementwise_add_op_desc.SetInput("Y", {fc_output_names[2]});
    elementwise_add_op_desc.SetOutput("Out", {elementwise_add_output_names[0]});

    elementwise_add_op_desc.Flush();
    VLOG(4) <<"set flush ok";
    auto* new_elementwise_add_op = graph->CreateOpNode(&elementwise_add_op_desc);
    VLOG(4) <<"create new_elementwise_add op node ok";

    IR_NODE_LINK_TO(fc_output_vars[0], new_elementwise_add_op);
    IR_NODE_LINK_TO(fc_output_vars[2], new_elementwise_add_op);
    IR_NODE_LINK_TO(new_elementwise_add_op, elementwise_add_output_vars[0]);


    // Create final split
    OpDesc split_op_desc;
    split_op_desc.SetType("split");
    
    std::vector<std::string> new_split_output_names;
    for(int j = 0;j<num_columns;j++){
      new_split_output_names.push_back(output_elementwise_add_vars_names[j]);  //output_elementwise_add_vars_names: original elementwise_add outputs
    }
    //新建split，改split的输入输出名字
    split_op_desc.SetInput("X", {elementwise_add_output_names[0]});
    split_op_desc.SetOutput("Out", new_split_output_names);
    //split_op_desc.SetAttr("axis", node_split_op->Op()->GetAttr("axis"));
    split_op_desc.SetAttr("axis", 0);
    std::vector<int> tmp_section;
    split_op_desc.SetAttr("sections", tmp_section);
    split_op_desc.SetAttr("num", node_split_op->Op()->GetAttr("num"));
    auto* new_split_op = graph->CreateOpNode(&split_op_desc);
    VLOG(4) <<"set new split op ok";
    IR_NODE_LINK_TO(elementwise_add_output_vars[0], new_split_op);


    for(int j = 0;j<num_columns;j++){
      output_elementwise_add_vars[j]->inputs.clear();
      IR_NODE_LINK_TO(new_split_op, output_elementwise_add_vars[j]);
    }

    VLOG(4) <<"begin_to_erase";
    std::unordered_set<const Node*> marked_nodes;
    for (auto& item : subgraph) {
      marked_nodes.insert(item.second);
    }

    marked_nodes.erase(node_split_input_var);
    marked_nodes.erase(elementwise_add_output_vars[0]);

    for(int i = 0;i< num_layers;i++)
    {
      marked_nodes.erase(fc_output_vars[i]);//newly add tmp var
    }
    for(int i = 0; i < num_layers; i++)
    {
      marked_nodes.erase(input_vars_w[i * num_columns]);
      marked_nodes.erase(input_vars_bias[i * num_columns]);
      //marked_nodes.erase(input_vars_input[i * num_columns]);
    }
    for(int j = 0; j < num_columns; j++)
    {
      marked_nodes.erase(output_elementwise_add_vars[j]);
    }

    VLOG(3) << "going to remoe nodes";
    for(auto* item:marked_nodes)
    {
      VLOG(3) <<item->Name();
    }
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;

  };

  gpd(graph, handler);
  return fusion_count;
}

void BatchedGemmFusePass_v2::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  int fusion_count = 0;
  for (int i = MAX_COLUMNS; i > 2; --i) {
      fusion_count +=
        BuildFusion_v2(graph, name_scope_ + "/" + std::to_string(i), i);
      VLOG(3) << "column:" + std::to_string(i) + " fusion_count:" + std::to_string(fusion_count);
  }
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(batched_gemm_fuse_pass_v2,
              paddle::framework::ir::BatchedGemmFusePass_v2);
