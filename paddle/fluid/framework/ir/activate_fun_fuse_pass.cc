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

#include "paddle/fluid/framework/ir/activate_fun_fuse_pass.h"
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

void BuildActivateFunFusePattern(PDPattern* pattern,
                                  const std::string& name_scope,
                                  int num_columns) {
  VLOG(4) << "in BuildActivateFunFusePattern";
  auto is_split_op_with_outputs = [](Node* x, int num) -> bool {
    return x && x->IsOp() && x->Op()->Type() == "split" &&
           x->Op()->Output("Out").size() == static_cast<size_t>(num);
  };

  auto is_nth_output_var_of_split = [=](Node* x, int idx) -> bool {
    return x && x->IsVar() && VarLinksFromOp(x, "split") && x->inputs.size() >= 1 &&
           x->outputs.size() == 1 && IsNthOutput(x, x->inputs[0], "Out", idx);
  };
  
  //support tanh or sigmoid
  auto is_nth_output_activate_op_of_split = [=](Node* x, int idx) -> bool {
    return x && x->IsOp() && ((x->Op()->Type() == "tanh")||(x->Op()->Type() == "sigmoid")) && x->inputs.size() >= 1 &&
           is_nth_output_var_of_split(x->inputs[0], idx);
  };
  
  auto is_input_var_of_split = [=](Node* x) -> bool {
    return x && x->IsVar() && VarLinksToOp(x, "split");
  };

  auto is_input_var_of_activate = [=](Node* x) -> bool {
    return x && x->IsVar() && (VarLinksToOp(x, "tanh")||VarLinksToOp(x, "sigmoid"));
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
  
  VLOG(4) << "find split";
  std::vector<PDNode*> activate_ops(num_columns);
  std::vector<PDNode*> activate_output_var(num_columns);
  std::vector<PDNode*> activate_input_var(num_columns);
  
  for (int i = 0; i < num_columns; ++i) {
    
    activate_ops[i] = pattern->NewNode(
        [=](Node* x) {
          return x && x->IsOp() && is_nth_output_activate_op_of_split(x, i);
        },
        name_scope + "/activate_op_" + std::to_string(i));
    VLOG(4) << "find avtivate op";

    activate_input_var[i] = pattern->NewNode(
        [=](Node* x) {
          bool basic = x && x->IsVar() && is_input_var_of_activate(x) && x->inputs.size() >= 1 &&
                       is_nth_output_var_of_split(x, i);
          return basic;
        },
        name_scope + "/split_output_" + std::to_string(i))->assert_is_op_output("split", "Out")->AsIntermediate();//also avtivate input
    VLOG(4) << "find split output var";
     activate_output_var[i] = pattern->NewNode(
        [=](Node* x) {
          bool basic = x && x->IsVar() && x->inputs.size() >= 1 && x->inputs[0]->IsOp() && 
          ((x->inputs[0]->Op()->Type() == "tanh")||(x->inputs[0]->Op()->Type() == "sigmoid")) && is_nth_output_var_of_split(x->inputs[0]->inputs[0], i);
          return basic;
        },
        name_scope + "/activate_output" + std::to_string(i))->AsOutput();
    VLOG(4) << "find activate output var";
    split_op->LinksTo({activate_input_var[i]});
    activate_ops[i]
      ->LinksFrom({activate_input_var[i]})
      .LinksTo({activate_output_var[i]});
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
  BuildActivateFunFusePattern(pattern, name_scope, num_columns);

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
    VLOG(4) << "handle ActivateFun fuse";
    std::vector<std::string> activate_ops_names(num_columns);
    std::vector<std::string> activate_input_vars_names(num_columns);
    std::vector<std::string> activate_output_vars_names(num_columns);
    
    std::vector<Node*> activate_ops(num_columns);
    std::vector<Node*> activate_input_vars(num_columns);
    std::vector<Node*> activate_output_vars(num_columns);

    auto& fused_pattern = gpd.pattern();

    Node* node_split_input_var =  retrieve_node(name_scope + "/split_input", subgraph, fused_pattern);
    std::string node_split_input_var_name = node_split_input_var->Name();
    Node* node_split_op =  retrieve_node(name_scope + "/split_op", subgraph, fused_pattern);

    for(int j = 0; j < num_columns; j++){

      activate_ops[j] =
        retrieve_node(name_scope + "/activate_op_" + std::to_string(j),
                      subgraph, fused_pattern);
      activate_ops_names[j] = activate_ops[j]->Name();

      activate_input_vars[j] =
        retrieve_node(name_scope + "/split_output_" + std::to_string(j),
                      subgraph, fused_pattern);
      activate_input_vars_names[j] = activate_input_vars[j]->Name();

      activate_output_vars[j] =
        retrieve_node(name_scope + "/activate_output" + std::to_string(j),
                      subgraph, fused_pattern);
      activate_output_vars_names[j] = activate_output_vars[j]->Name();

    }
    
    VLOG(4) << "my_special_flag";
    VLOG(4) << node_split_input_var_name;  
    for(auto item:activate_ops_names)
       VLOG(4) << item; 
    for(auto item:activate_input_vars_names)
       VLOG(4) << item;        
    for(auto item:activate_output_vars_names)
       VLOG(4) << item; 

    //get shape
    //std::vector<int64_t> GetShape
    std::vector<int64_t> new_activate_shape = node_split_input_var->Var()->GetShape();
    const std::string& new_activate_out_var_name = patterns::UniqueKey("new_activate_out");
    auto* new_activate_out_var_desc =
        activate_ops[0]->Op()->Block()->Var(new_activate_out_var_name);//for var can be get from scope
    new_activate_out_var_desc->SetPersistable(false);
    new_activate_out_var_desc->SetShape(new_activate_shape);
    new_activate_out_var_desc->SetDataType(node_split_input_var->Var()->GetDataType());
    auto* new_activate_out_var = g->CreateVarNode(new_activate_out_var_desc);
    VLOG(4) <<"create new activate out var ok";

    node_split_input_var->outputs.clear();
    // combines activate op
    OpDesc activate_op_desc(activate_ops[0]->Op()->Block());
    activate_op_desc.SetType(activate_ops[0]->Op()->Type());
    activate_op_desc.SetInput("X", {node_split_input_var_name});
    activate_op_desc.SetOutput("Out", {new_activate_out_var_name});

    activate_op_desc.SetAttr("support_int8", activate_ops[0]->Op()->GetAttr("support_int8"));
    activate_op_desc.SetAttr("use_cudnn", activate_ops[0]->Op()->GetAttr("use_cudnn"));
    activate_op_desc.SetAttr("use_mkldnn", activate_ops[0]->Op()->GetAttr("use_mkldnn"));
    Node* node_activate_op = graph->CreateOpNode(&activate_op_desc);
    
    IR_NODE_LINK_TO(node_split_input_var, node_activate_op);
    IR_NODE_LINK_TO(node_activate_op, new_activate_out_var);

    OpDesc split_op_desc(activate_ops[0]->Op()->Block());
    split_op_desc.SetType("split");
    //新建split，改split的输入输出名字
    split_op_desc.SetInput("X", {new_activate_out_var_name});
    split_op_desc.SetOutput("Out", activate_output_vars_names);
    std::vector<int> tmp_section;
    split_op_desc.SetAttr("sections", tmp_section);
    split_op_desc.SetAttr("axis", node_split_op->Op()->GetAttr("axis"));
    split_op_desc.SetAttr("num", node_split_op->Op()->GetAttr("num"));
    auto* new_split_op = graph->CreateOpNode(&split_op_desc);
    VLOG(4) <<"set new split op ok";
    IR_NODE_LINK_TO(new_activate_out_var, new_split_op);

    for(int j = 0;j<num_columns;j++){
      activate_output_vars[j]->inputs.clear();
      IR_NODE_LINK_TO(new_split_op, activate_output_vars[j]);
    }
      
    VLOG(4) <<"begin_to_erase";//erase means to keep them
    std::unordered_set<const Node*> marked_nodes;
    for (auto& item : subgraph) {
      marked_nodes.insert(item.second);
    }
    marked_nodes.insert(node_split_op);
    marked_nodes.erase(node_activate_op);
    marked_nodes.erase(node_split_input_var);
    for(int i = 0;i< num_columns;i++)
    {
      marked_nodes.erase(activate_output_vars[i]);
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

void ActivateFunFusePass::ApplyImpl(ir::Graph* graph) const {
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

REGISTER_PASS(activate_fun_fuse_pass,
              paddle::framework::ir::ActivateFunFusePass);
