// code/graph/graph.h — 计算图层总入口：ActiveModel -> 它的层执行器
//
// 编译期完成路由，运行期没有任何查表 / 分支。Engine::layer() 只剩这一句：
//     graph::ActiveGraph::run(ctx);
#pragma once
#include "graph/spec.h"
#include "graph/dense.h"
#include "graph/qwen2.h"
#include "graph/llama.h"

namespace nova {
namespace graph {

using ActiveSpec  = GraphSpec<Model>;   // Model = nova::ActiveModel（见 nova/kernel.h）
using ActiveGraph = GraphFor<Model>;

// 计算图层与模具必须同源：任何一层写歪了都在这里编译期炸
static_assert(ActiveGraph::spec::D == Model::D, "graph spec 与模具 D 不一致");
static_assert(ActiveGraph::spec::L == Model::L, "graph spec 与模具 L 不一致");
static_assert(ActiveGraph::spec::tied == Model::tied, "graph spec 与模具 tie 不一致");
static_assert(ActiveGraph::n_ops > 0, "层序列不能为空");

} // namespace graph
} // namespace nova
