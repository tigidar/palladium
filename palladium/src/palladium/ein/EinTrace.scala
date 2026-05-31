package palladium.ein

import scala.collection.mutable

enum EinNodeKind:
  case Parameter(name: String)
  case Input(name: String)
  case Fill
  case Contract
  case ElemAdd
  case ElemSub
  case ElemMul
  case Activate(f: Activation)
  case ActivateDeriv(f: Activation)
  case ReduceSum(over: List[String])
  case Broadcast
  case Transpose
  case Softmax(overDim: String)
  case LogSoftmax(overDim: String)
  case LayerNorm(overDims: List[String], eps: Double)
  case Reshape(targetDims: List[Dim])
  case Slice(dim: String, from: Int, to: Int)
  case Gather(lookupDim: String, indexDims: List[Dim])
  case Scatter(lookupDim: String, tableDims: List[Dim])

case class EinTracedNode(
    id: Int,
    label: String,
    kind: EinNodeKind,
    outputDims: List[Dim],
    inputs: List[Int]
)

case class EinTracedGraph(
    nodes: Vector[EinTracedNode],
    rootId: Int
)

object EinTrace:

  def forward[A](expr: Ein[A]): EinTracedGraph =
    val nodes = mutable.ArrayBuffer.empty[EinTracedNode]
    val paramMap = mutable.Map.empty[String, Int]

    def walk(e: Ein[A]): Int =
      e match
        case Ein.Param(id, dims, _) =>
          paramMap.getOrElseUpdate(
            id, {
              val nid = nodes.size
              nodes += EinTracedNode(nid, id, EinNodeKind.Parameter(id), dims, Nil)
              nid
            }
          )

        case Ein.Input(id, dims) =>
          paramMap.getOrElseUpdate(
            s"input:$id", {
              val nid = nodes.size
              nodes += EinTracedNode(nid, id, EinNodeKind.Input(id), dims, Nil)
              nid
            }
          )

        case Ein.Fill(_, dims) =>
          val nid = nodes.size
          nodes += EinTracedNode(nid, "fill", EinNodeKind.Fill, dims, Nil)
          nid

        case Ein.Ones(dims) =>
          val nid = nodes.size
          nodes += EinTracedNode(nid, "ones", EinNodeKind.Fill, dims, Nil)
          nid

        case Ein.Zeros(dims) =>
          val nid = nodes.size
          nodes += EinTracedNode(nid, "zeros", EinNodeKind.Fill, dims, Nil)
          nid

        case Ein.Contract(left, right) =>
          val lId = walk(left)
          val rId = walk(right)
          val nid = nodes.size
          val dims = e.outputDims
          nodes += EinTracedNode(nid, "contract", EinNodeKind.Contract, dims, List(lId, rId))
          nid

        case Ein.ElemAdd(left, right) =>
          val lId = walk(left)
          val rId = walk(right)
          val nid = nodes.size
          val dims = e.outputDims
          nodes += EinTracedNode(nid, "+", EinNodeKind.ElemAdd, dims, List(lId, rId))
          nid

        case Ein.ElemSub(left, right) =>
          val lId = walk(left)
          val rId = walk(right)
          val nid = nodes.size
          val dims = e.outputDims
          nodes += EinTracedNode(nid, "-", EinNodeKind.ElemSub, dims, List(lId, rId))
          nid

        case Ein.ElemMul(left, right) =>
          val lId = walk(left)
          val rId = walk(right)
          val nid = nodes.size
          val dims = e.outputDims
          nodes += EinTracedNode(nid, ".*", EinNodeKind.ElemMul, dims, List(lId, rId))
          nid

        case Ein.Activate(f, arg) =>
          val aId = walk(arg)
          val nid = nodes.size
          val label = f match
            case Activation.ReLU    => "ReLU"
            case Activation.Sigmoid => "Sigmoid"
            case Activation.Tanh    => "Tanh"
            case Activation.GELU    => "GELU"
            case Activation.Swish   => "Swish"
          nodes += EinTracedNode(nid, label, EinNodeKind.Activate(f), e.outputDims, List(aId))
          nid

        case Ein.ActivateDeriv(f, arg) =>
          val aId = walk(arg)
          val nid = nodes.size
          val label = f match
            case Activation.ReLU    => "ReLU'"
            case Activation.Sigmoid => "Sigmoid'"
            case Activation.Tanh    => "Tanh'"
            case Activation.GELU    => "GELU'"
            case Activation.Swish   => "Swish'"
          nodes += EinTracedNode(nid, label, EinNodeKind.ActivateDeriv(f), e.outputDims, List(aId))
          nid

        case Ein.ReduceSum(arg, over) =>
          val aId = walk(arg)
          val nid = nodes.size
          nodes += EinTracedNode(nid, s"sum(${over.mkString(",")})", EinNodeKind.ReduceSum(over), e.outputDims, List(aId))
          nid

        case Ein.Broadcast(arg, targetDims) =>
          val aId = walk(arg)
          val nid = nodes.size
          nodes += EinTracedNode(nid, "broadcast", EinNodeKind.Broadcast, targetDims, List(aId))
          nid

        case Ein.Transpose(arg, _) =>
          val aId = walk(arg)
          val nid = nodes.size
          nodes += EinTracedNode(nid, "transpose", EinNodeKind.Transpose, e.outputDims, List(aId))
          nid

        case Ein.Softmax(arg, overDim) =>
          val aId = walk(arg)
          val nid = nodes.size
          nodes += EinTracedNode(nid, s"softmax($overDim)", EinNodeKind.Softmax(overDim), e.outputDims, List(aId))
          nid

        case Ein.LogSoftmax(arg, overDim) =>
          val aId = walk(arg)
          val nid = nodes.size
          nodes += EinTracedNode(nid, s"log_softmax($overDim)", EinNodeKind.LogSoftmax(overDim), e.outputDims, List(aId))
          nid

        case Ein.LayerNorm(arg, scale, bias, overDims, eps) =>
          val argId = walk(arg)
          val scaleId = walk(scale)
          val biasId = walk(bias)
          val nid = nodes.size
          nodes += EinTracedNode(nid, s"layernorm(${overDims.mkString(",")})", EinNodeKind.LayerNorm(overDims, eps), e.outputDims, List(argId, scaleId, biasId))
          nid

        case Ein.Reshape(arg, targetDims) =>
          val aId = walk(arg)
          val nid = nodes.size
          nodes += EinTracedNode(nid, "reshape", EinNodeKind.Reshape(targetDims), e.outputDims, List(aId))
          nid

        case Ein.Slice(arg, dim, from, to) =>
          val aId = walk(arg)
          val nid = nodes.size
          nodes += EinTracedNode(nid, s"slice($dim,$from:$to)", EinNodeKind.Slice(dim, from, to), e.outputDims, List(aId))
          nid

        case Ein.Gather(table, indices, lookupDim) =>
          val tId = walk(table)
          val nid = nodes.size
          nodes += EinTracedNode(nid, s"gather($lookupDim)", EinNodeKind.Gather(lookupDim, indices.dims), e.outputDims, List(tId))
          nid

        case Ein.Scatter(src, indices, lookupDim, tableDims) =>
          val sId = walk(src)
          val nid = nodes.size
          nodes += EinTracedNode(nid, s"scatter($lookupDim)", EinNodeKind.Scatter(lookupDim, tableDims), e.outputDims, List(sId))
          nid

    val rootId = walk(expr)
    EinTracedGraph(nodes.toVector, rootId)

  object syntax:
    extension [A](expr: Ein[A])
      def toGraph: EinTracedGraph = EinTrace.forward(expr)
