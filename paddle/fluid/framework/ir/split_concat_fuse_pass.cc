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

#include "paddle/fluid/framework/ir/split_concat_fuse_pass.h"
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

void BuildSplitConcatFusePattern(PDPattern* pattern,
                                  const std::string& name_scope,
                                  int num_columns) {
  VLOG(4) << "in BuildSplitConcatFunFusePattern";
  auto is_split_op_with_outputs = [](Node* x, int num) -> bool {
    return x && x->IsOp() && x->Op()->Type() == "split" &&  BOOST_GET_CONST(int, x->Op()->GetAttr("axis")) == int(1) &&
           x->Op()->Output("Out").size() == static_cast<size_t>(num);
  };
  auto is_concat_op_with_inputs = [](Node* x, int num) -> bool {
    return x && x->IsOp() && x->Op()->Type() == "concat" && BOOST_GET_CONST(int, x->Op()->GetAttr("axis")) == int(1) && //only support axis==1
           x->Op()->Input("X").size() == static_cast<size_t>(num);
  };
  auto is_nth_output_var_of_split = [=](Node* x, int idx) -> bool {
    return x && x->IsVar() && VarLinksFromOp(x, "split") && x->inputs.size() >= 1 &&
           x->outputs.size() == 1 && IsNthOutput(x, x->inputs[0], "Out", idx);
  };
  auto is_nth_input_var_of_concat = [=](Node* x, int idx) -> bool {
    return x && x->IsVar() && VarLinksToOp(x, "concat") && x->inputs.size() >= 1 &&
            x->outputs.size() == 1 && IsNthInput(x, x->outputs[0], "X", idx);
  };
  auto is_input_var_of_split = [=](Node* x) -> bool {
    return x && x->IsVar() && VarLinksToOp(x, "split");
  };

  //limit in_op in [tanh, sigmoid]
  PDNode* in_op = pattern->NewNode(
      [=](Node* x) { 
        bool basic = x && x->IsOp() && ((x->Op()->Type() == "tanh")||(x->Op()->Type() == "sigmoid"));
        return basic;
       },
      name_scope + "/in_op")->AsInput();

  PDNode* split_input_var = pattern->NewNode(
      [=](Node* x) { 
        bool basic = x && x->IsVar() && is_input_var_of_split(x);
        return basic;
       },
      name_scope + "/split_input")->assert_is_op_input("split", "X");

  PDNode* split_op = pattern->NewNode(
      [=](Node* x) { return is_split_op_with_outputs(x, num_columns); },
      name_scope + "/split_op");

  PDNode* concat_op = pattern->NewNode(
      [=](Node* x) { return is_concat_op_with_inputs(x, num_columns); },
      name_scope + "/concat_op");

  PDNode* concat_output_var = pattern->NewNode(
      [=](Node* x) { 
        bool basic = x && x->IsVar() && VarLinksFromOp(x, "concat");
        return basic;
       },
      name_scope + "/concat_output")->assert_is_op_output("concat", "Out");
  
  PDNode* out_op = pattern->NewNode(
      [=](Node* x) { 
        bool basic = x && x->IsOp() && x->inputs.size() ==1 && VarLinksFromOp(x->inputs[0], "concat");
        return basic;
       },
      name_scope + "/out_op")->AsOutput();

  in_op->LinksTo({split_input_var});
  split_op->LinksFrom({split_input_var});
  concat_op->LinksTo({concat_output_var});
  out_op->LinksFrom({concat_output_var});
  VLOG(4) << "find split and concat";
  
  std::vector<PDNode*> split_output_var(num_columns);
  for (int i = 0; i < num_columns; ++i) {
    split_output_var[i] = pattern->NewNode(
        [=](Node* x) {
          bool basic = x && x->IsVar() &&  x->inputs.size() == 1 && is_nth_output_var_of_split(x, i) && is_nth_input_var_of_concat(x, i);
          return basic;
        },
        name_scope + "/split_output_" + std::to_string(i))->assert_is_op_output("split", "Out")->AsIntermediate();//also concat input

    split_op->LinksTo({split_output_var[i]});
    concat_op->LinksFrom({split_output_var[i]});
  }

  VLOG(4) << "finish";
  return;
}

static int BuildFusion(Graph* graph, const std::string& name_scope,
                       int num_columns) {
  //get scpoe
  PADDLE_ENFORCE_EQ(graph->Has(kParamScopeAttr), true,
                    platform::errors::InvalidArgument(
                        "Graph must have kParamScopeAttr attribute."));
  //auto& scope = graph->Get<Scope>(kParamScopeAttr);
  
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  VLOG(3) << "pattern string " + pattern->DotString();
  BuildSplitConcatFusePattern(pattern, name_scope, num_columns);

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
    VLOG(4) << "handle SplitConcat fuse";
    std::vector<std::string> split_output_vars_names(num_columns);
    std::vector<Node*> split_output_vars(num_columns);

    auto& fused_pattern = gpd.pattern();

    Node* node_split_input_var =  retrieve_node(name_scope + "/split_input", subgraph, fused_pattern);
    std::string node_split_input_var_name = node_split_input_var->Name();
    //Node* node_split_op =  retrieve_node(name_scope + "/split_op", subgraph, fused_pattern);
    Node* node_in_op =  retrieve_node(name_scope + "/in_op", subgraph, fused_pattern);
    Node* node_out_op =  retrieve_node(name_scope + "/out_op", subgraph, fused_pattern);

    Node* node_concat_output_var =  retrieve_node(name_scope + "/concat_output", subgraph, fused_pattern);
    std::string node_concat_output_var_name = node_concat_output_var->Name();
    Node* node_concat_op =  retrieve_node(name_scope + "/concat_op", subgraph, fused_pattern);

    for(int j = 0; j < num_columns; j++){
      split_output_vars[j] =
        retrieve_node(name_scope + "/split_output_" + std::to_string(j),
                      subgraph, fused_pattern);
      split_output_vars_names[j] = split_output_vars[j]->Name();
    }
    
    VLOG(4) << "my_special_flag";
    VLOG(4) << node_split_input_var_name;
    for(auto item:split_output_vars_names)
       VLOG(4) << item; 

    //begin to handle
    std::vector<int64_t> new_var_shape = node_split_input_var->Var()->GetShape();
    const std::string& new_var_name = patterns::UniqueKey("new_var");
    auto* new_var_desc =
        node_concat_op->Op()->Block()->Var(new_var_name);//for var can be get from scope
    new_var_desc->SetPersistable(false);
    new_var_desc->SetShape(new_var_shape);
    new_var_desc->SetDataType(node_split_input_var->Var()->GetDataType());
    auto* new_var = g->CreateVarNode(new_var_desc);
    VLOG(4) <<"create new var ok";

    node_in_op->outputs.clear();
    node_out_op->inputs.clear();
    IR_NODE_LINK_TO(node_in_op, new_var);
    IR_NODE_LINK_TO(new_var, node_out_op);

    //modify node_in_op's outputs
    auto* node_in_op_desc = node_in_op->Op();
    auto var_map = node_in_op_desc->Outputs();
    for (auto& name_m : var_map) {
      if (std::find(name_m.second.begin(),
                    name_m.second.end(),
                    node_split_input_var_name) != name_m.second.end()) {
        std::vector<std::string> new_outputs;
        for (auto& i_n : name_m.second) {
          if (i_n != node_split_input_var_name) {
            new_outputs.push_back(i_n);
          }
        }
        new_outputs.push_back(new_var_name);
        node_in_op_desc->SetOutput(name_m.first, new_outputs);
        node_in_op_desc->Flush();
      }
    }
    node_in_op_desc->Flush();


    //modify node_out_op's inputs
    auto* node_out_op_desc = node_out_op->Op();
    auto var_map_out = node_out_op_desc->Inputs();
    for (auto& name_m : var_map_out) {
      if (std::find(name_m.second.begin(),
                    name_m.second.end(),
                    node_concat_output_var_name) != name_m.second.end()) {
        std::vector<std::string> new_inputs;
        for (auto& i_n : name_m.second) {
          if (i_n != node_concat_output_var_name) {
            new_inputs.push_back(i_n);
          }
        }
        new_inputs.push_back(new_var_name);
        node_out_op_desc->SetInput(name_m.first, new_inputs);
        node_out_op_desc->Flush();
      }
    }
    node_out_op_desc->Flush();

    VLOG(4) <<"begin_to_erase";//erase means to keep them
    std::unordered_set<const Node*> marked_nodes;
    for (auto& item : subgraph) {
      marked_nodes.insert(item.second);
    }
    for(int j = 0; j < num_columns; j++){
      marked_nodes.insert(split_output_vars[j]);
    }
    marked_nodes.erase(node_in_op);
    marked_nodes.erase(node_out_op);

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

void SplitConcatFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  int fusion_count = 0;
  for (int i = MAX_COLUMNS; i > 2; --i) {
      fusion_count +=
        BuildFusion(graph, name_scope_ + "/" + std::to_string(i), i);
      VLOG(3) << "column:" + std::to_string(i) + " fusion_count:" + std::to_string(fusion_count);
  }
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(split_concat_fuse_pass,
              paddle::framework::ir::SplitConcatFusePass);
