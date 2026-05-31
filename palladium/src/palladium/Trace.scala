package palladium

import scala.collection.mutable

enum OpType:
  case Add, Sub, Mul, Div, FastDiv, Pow, Neg, Log, Exp, Tanh, Sigmoid, Relu, Sin, Cos, Abs, Step, Signum

enum NodeKind:
  case Variable(name: String)
  case Literal
  case Constant
  case Op(op: OpType)

case class TracedNode[A](
    id: Int,
    label: String,
    kind: NodeKind,
    value: A,
    inputs: List[Int]
)

case class TracedGraph[A](
    nodes: Vector[TracedNode[A]],
    rootId: Int
)

object Trace:

  /** Walk the Value[A] tree, evaluate at every node, deduplicate Vars, produce a flat DAG. */
  def forward[A: NumberLike](expr: Value[A]): TracedGraph[A] =
    val num = summon[NumberLike[A]]
    val nodes = mutable.ArrayBuffer.empty[TracedNode[A]]
    val varMap = mutable.Map.empty[String, Int]

    def walk(v: Value[A]): Int =
      import Value.*
      v match
        case Var(id, data) =>
          varMap.getOrElseUpdate(
            id, {
              val nid = nodes.size
              nodes += TracedNode(nid, id, NodeKind.Variable(id), data, Nil)
              nid
            }
          )

        case Lit(data) =>
          val nid = nodes.size
          nodes += TracedNode(nid, data.toString, NodeKind.Literal, data, Nil)
          nid

        case Const(n) =>
          val nid = nodes.size
          nodes += TracedNode(
            nid,
            n.toString,
            NodeKind.Constant,
            num.fromInt(n),
            Nil
          )
          nid

        case Add(l, r) =>
          val lId = walk(l)
          val rId = walk(r)
          val nid = nodes.size
          val value = num.plus(nodes(lId).value, nodes(rId).value)
          nodes += TracedNode(
            nid,
            "+",
            NodeKind.Op(OpType.Add),
            value,
            List(lId, rId)
          )
          nid

        case Sub(l, r) =>
          val lId = walk(l)
          val rId = walk(r)
          val nid = nodes.size
          val value = num.minus(nodes(lId).value, nodes(rId).value)
          nodes += TracedNode(
            nid,
            "-",
            NodeKind.Op(OpType.Sub),
            value,
            List(lId, rId)
          )
          nid

        case Mul(l, r) =>
          val lId = walk(l)
          val rId = walk(r)
          val nid = nodes.size
          val value = num.times(nodes(lId).value, nodes(rId).value)
          nodes += TracedNode(
            nid,
            "*",
            NodeKind.Op(OpType.Mul),
            value,
            List(lId, rId)
          )
          nid

        case Div(l, r) =>
          val lId = walk(l)
          val rId = walk(r)
          val nid = nodes.size
          val value = num.div(nodes(lId).value, nodes(rId).value)
          nodes += TracedNode(
            nid,
            "/",
            NodeKind.Op(OpType.Div),
            value,
            List(lId, rId)
          )
          nid

        case FastDiv(l, r) =>
          val lId = walk(l)
          val rId = walk(r)
          val nid = nodes.size
          val rVal = nodes(rId).value
          val value = num.times(nodes(lId).value, num.pow(rVal, num.negate(num.fromInt(1))))
          nodes += TracedNode(
            nid,
            "/~",
            NodeKind.Op(OpType.FastDiv),
            value,
            List(lId, rId)
          )
          nid

        case Pow(base, exp) =>
          val bId = walk(base)
          val eId = walk(exp)
          val nid = nodes.size
          val value = num.pow(nodes(bId).value, nodes(eId).value)
          nodes += TracedNode(
            nid,
            "^",
            NodeKind.Op(OpType.Pow),
            value,
            List(bId, eId)
          )
          nid

        case Neg(v) =>
          val cId = walk(v)
          val nid = nodes.size
          val value = num.negate(nodes(cId).value)
          nodes += TracedNode(
            nid,
            "neg",
            NodeKind.Op(OpType.Neg),
            value,
            List(cId)
          )
          nid

        case Log(v) =>
          val cId = walk(v)
          val nid = nodes.size
          val value = num.log(nodes(cId).value)
          nodes += TracedNode(
            nid,
            "log",
            NodeKind.Op(OpType.Log),
            value,
            List(cId)
          )
          nid

        case Exp(v) =>
          val cId = walk(v)
          val nid = nodes.size
          val value = num.exp(nodes(cId).value)
          nodes += TracedNode(
            nid,
            "exp",
            NodeKind.Op(OpType.Exp),
            value,
            List(cId)
          )
          nid

        case Tanh(v) =>
          val cId = walk(v)
          val nid = nodes.size
          val vVal = nodes(cId).value
          val ev = num.exp(vVal)
          val emv = num.exp(num.negate(vVal))
          val value = num.div(num.minus(ev, emv), num.plus(ev, emv))
          nodes += TracedNode(
            nid,
            "tanh",
            NodeKind.Op(OpType.Tanh),
            value,
            List(cId)
          )
          nid

        case Sigmoid(v) =>
          val cId = walk(v)
          val nid = nodes.size
          val vVal = nodes(cId).value
          // σ(x) = 1 / (1 + exp(-x))
          val value = num.div(num.fromInt(1), num.plus(num.fromInt(1), num.exp(num.negate(vVal))))
          nodes += TracedNode(
            nid,
            "σ",
            NodeKind.Op(OpType.Sigmoid),
            value,
            List(cId)
          )
          nid

        case Relu(v) =>
          val cId = walk(v)
          val nid = nodes.size
          val value = num.relu(nodes(cId).value)
          nodes += TracedNode(
            nid,
            "relu",
            NodeKind.Op(OpType.Relu),
            value,
            List(cId)
          )
          nid

        case Sin(v) =>
          val cId = walk(v)
          val nid = nodes.size
          val value = num.sin(nodes(cId).value)
          nodes += TracedNode(
            nid,
            "sin",
            NodeKind.Op(OpType.Sin),
            value,
            List(cId)
          )
          nid

        case Cos(v) =>
          val cId = walk(v)
          val nid = nodes.size
          val value = num.cos(nodes(cId).value)
          nodes += TracedNode(
            nid,
            "cos",
            NodeKind.Op(OpType.Cos),
            value,
            List(cId)
          )
          nid

        case Abs(v) =>
          val cId = walk(v)
          val nid = nodes.size
          val value = num.abs(nodes(cId).value)
          nodes += TracedNode(
            nid,
            "|·|",
            NodeKind.Op(OpType.Abs),
            value,
            List(cId)
          )
          nid

        case Step(v) =>
          val cId = walk(v)
          val nid = nodes.size
          val value = num.step(nodes(cId).value)
          nodes += TracedNode(
            nid,
            "step",
            NodeKind.Op(OpType.Step),
            value,
            List(cId)
          )
          nid

        case Signum(v) =>
          val cId = walk(v)
          val nid = nodes.size
          val value = num.signum(nodes(cId).value)
          nodes += TracedNode(
            nid,
            "sgn",
            NodeKind.Op(OpType.Signum),
            value,
            List(cId)
          )
          nid

    val rootId = walk(expr)
    TracedGraph(nodes.toVector, rootId)

  /** Reverse-mode backward pass over the traced DAG. Returns gradient at every node. */
  def backward[A: NumberLike](graph: TracedGraph[A]): Map[Int, A] =
    val num = summon[NumberLike[A]]
    val zero = num.fromInt(0)
    val grads = mutable.Map[Int, A]()
    for node <- graph.nodes do grads(node.id) = zero
    grads(graph.rootId) = num.fromInt(1)

    // Nodes are in topological order (children before parents),
    // so reverse iteration gives correct backward ordering.
    for node <- graph.nodes.reverseIterator do
      val upstream = grads(node.id)
      node.kind match
        case NodeKind.Op(OpType.Add) =>
          val List(lId, rId) = node.inputs: @unchecked
          grads(lId) = num.plus(grads(lId), upstream)
          grads(rId) = num.plus(grads(rId), upstream)

        case NodeKind.Op(OpType.Sub) =>
          val List(lId, rId) = node.inputs: @unchecked
          grads(lId) = num.plus(grads(lId), upstream)
          grads(rId) = num.plus(grads(rId), num.negate(upstream))

        case NodeKind.Op(OpType.Mul) =>
          val List(lId, rId) = node.inputs: @unchecked
          grads(lId) =
            num.plus(grads(lId), num.times(upstream, graph.nodes(rId).value))
          grads(rId) =
            num.plus(grads(rId), num.times(upstream, graph.nodes(lId).value))

        case NodeKind.Op(OpType.Div) =>
          val List(lId, rId) = node.inputs: @unchecked
          val lVal = graph.nodes(lId).value
          val rVal = graph.nodes(rId).value
          grads(lId) = num.plus(grads(lId), num.div(upstream, rVal))
          grads(rId) = num.plus(
            grads(rId),
            num.negate(
              num.div(num.times(upstream, lVal), num.times(rVal, rVal))
            )
          )

        case NodeKind.Op(OpType.FastDiv) =>
          // FastDiv(a, b) = a * b^(-1)
          // d/da = b^(-1), d/db = -a * b^(-2)
          val List(lId, rId) = node.inputs: @unchecked
          val lVal = graph.nodes(lId).value
          val rVal = graph.nodes(rId).value
          val rInv = num.pow(rVal, num.negate(num.fromInt(1)))
          grads(lId) = num.plus(grads(lId), num.times(upstream, rInv))
          grads(rId) = num.plus(
            grads(rId),
            num.negate(
              num.times(upstream, num.times(lVal, num.pow(rVal, num.negate(num.fromInt(2)))))
            )
          )

        case NodeKind.Op(OpType.Pow) =>
          val List(bId, eId) = node.inputs: @unchecked
          val baseVal = graph.nodes(bId).value
          val expVal = graph.nodes(eId).value
          val baseGrad = num.times(
            upstream,
            num.times(
              expVal,
              num.pow(baseVal, num.minus(expVal, num.fromInt(1)))
            )
          )
          grads(bId) = num.plus(grads(bId), baseGrad)
          val expGrad = num.times(
            upstream,
            num.times(num.pow(baseVal, expVal), num.log(baseVal))
          )
          grads(eId) = num.plus(grads(eId), expGrad)

        case NodeKind.Op(OpType.Neg) =>
          val List(cId) = node.inputs: @unchecked
          grads(cId) = num.plus(grads(cId), num.negate(upstream))

        case NodeKind.Op(OpType.Log) =>
          val List(cId) = node.inputs: @unchecked
          grads(cId) =
            num.plus(grads(cId), num.div(upstream, graph.nodes(cId).value))

        case NodeKind.Op(OpType.Exp) =>
          val List(cId) = node.inputs: @unchecked
          grads(cId) =
            num.plus(grads(cId), num.times(upstream, num.exp(graph.nodes(cId).value)))

        case NodeKind.Op(OpType.Tanh) =>
          val List(cId) = node.inputs: @unchecked
          val tanhVal = node.value
          grads(cId) =
            num.plus(grads(cId), num.times(upstream, num.minus(num.fromInt(1), num.times(tanhVal, tanhVal))))

        case NodeKind.Op(OpType.Sigmoid) =>
          // d/dv σ(v) = σ(v) * (1 - σ(v))
          val List(cId) = node.inputs: @unchecked
          val s = node.value
          grads(cId) =
            num.plus(grads(cId), num.times(upstream, num.times(s, num.minus(num.fromInt(1), s))))

        case NodeKind.Op(OpType.Relu) =>
          // d/dv relu(v) = step(v)
          val List(cId) = node.inputs: @unchecked
          grads(cId) =
            num.plus(grads(cId), num.times(upstream, num.step(graph.nodes(cId).value)))

        case NodeKind.Op(OpType.Sin) =>
          // d/dv sin(v) = cos(v)
          val List(cId) = node.inputs: @unchecked
          grads(cId) =
            num.plus(grads(cId), num.times(upstream, num.cos(graph.nodes(cId).value)))

        case NodeKind.Op(OpType.Cos) =>
          // d/dv cos(v) = -sin(v)
          val List(cId) = node.inputs: @unchecked
          grads(cId) =
            num.plus(grads(cId), num.times(upstream, num.negate(num.sin(graph.nodes(cId).value))))

        case NodeKind.Op(OpType.Abs) =>
          // d/dv |v| = sign(v)
          val List(cId) = node.inputs: @unchecked
          grads(cId) =
            num.plus(grads(cId), num.times(upstream, num.signum(graph.nodes(cId).value)))

        case NodeKind.Op(OpType.Step) | NodeKind.Op(OpType.Signum) =>
          ()

        case _ => () // leaf nodes — nothing to propagate

    grads.toMap

  /** Extract variable gradients from the full node-level gradient map. */
  def varGrads[A](graph: TracedGraph[A], grads: Map[Int, A]): Map[String, A] =
    graph.nodes.collect { case TracedNode(id, _, NodeKind.Variable(name), _, _) =>
      name -> grads(id)
    }.toMap

  /** Compute local derivatives for each edge (parentId, inputId) → local ∂parent/∂input. */
  def localDerivatives[A: NumberLike](graph: TracedGraph[A]): Map[(Int, Int), A] =
    val num = summon[NumberLike[A]]
    val result = scala.collection.mutable.Map.empty[(Int, Int), A]

    for node <- graph.nodes do
      node.kind match
        case NodeKind.Op(OpType.Add) =>
          val List(lId, rId) = node.inputs: @unchecked
          result((node.id, lId)) = num.fromInt(1)
          result((node.id, rId)) = num.fromInt(1)

        case NodeKind.Op(OpType.Sub) =>
          val List(lId, rId) = node.inputs: @unchecked
          result((node.id, lId)) = num.fromInt(1)
          result((node.id, rId)) = num.negate(num.fromInt(1))

        case NodeKind.Op(OpType.Mul) =>
          val List(lId, rId) = node.inputs: @unchecked
          result((node.id, lId)) = graph.nodes(rId).value
          result((node.id, rId)) = graph.nodes(lId).value

        case NodeKind.Op(OpType.Div) =>
          val List(lId, rId) = node.inputs: @unchecked
          val lVal = graph.nodes(lId).value
          val rVal = graph.nodes(rId).value
          result((node.id, lId)) = num.div(num.fromInt(1), rVal)
          result((node.id, rId)) = num.negate(
            num.div(lVal, num.times(rVal, rVal))
          )

        case NodeKind.Op(OpType.FastDiv) =>
          // FastDiv(a, b) = a * b^(-1)
          // d/da = b^(-1), d/db = -a * b^(-2)
          val List(lId, rId) = node.inputs: @unchecked
          val lVal = graph.nodes(lId).value
          val rVal = graph.nodes(rId).value
          result((node.id, lId)) = num.pow(rVal, num.negate(num.fromInt(1)))
          result((node.id, rId)) = num.negate(
            num.times(lVal, num.pow(rVal, num.negate(num.fromInt(2))))
          )

        case NodeKind.Op(OpType.Pow) =>
          val List(bId, eId) = node.inputs: @unchecked
          val baseVal = graph.nodes(bId).value
          val expVal = graph.nodes(eId).value
          result((node.id, bId)) = num.times(
            expVal,
            num.pow(baseVal, num.minus(expVal, num.fromInt(1)))
          )
          result((node.id, eId)) = num.times(
            num.pow(baseVal, expVal),
            num.log(baseVal)
          )

        case NodeKind.Op(OpType.Neg) =>
          val List(cId) = node.inputs: @unchecked
          result((node.id, cId)) = num.negate(num.fromInt(1))

        case NodeKind.Op(OpType.Log) =>
          val List(cId) = node.inputs: @unchecked
          result((node.id, cId)) = num.div(num.fromInt(1), graph.nodes(cId).value)

        case NodeKind.Op(OpType.Exp) =>
          val List(cId) = node.inputs: @unchecked
          result((node.id, cId)) = num.exp(graph.nodes(cId).value)

        case NodeKind.Op(OpType.Tanh) =>
          val List(cId) = node.inputs: @unchecked
          val tanhVal = node.value
          result((node.id, cId)) = num.minus(num.fromInt(1), num.times(tanhVal, tanhVal))

        case NodeKind.Op(OpType.Sigmoid) =>
          // d/dv σ(v) = σ(v) * (1 - σ(v))
          val List(cId) = node.inputs: @unchecked
          val s = node.value
          result((node.id, cId)) = num.times(s, num.minus(num.fromInt(1), s))

        case NodeKind.Op(OpType.Relu) =>
          // d/dv relu(v) = step(v)
          val List(cId) = node.inputs: @unchecked
          result((node.id, cId)) = num.step(graph.nodes(cId).value)

        case NodeKind.Op(OpType.Sin) =>
          // d/dv sin(v) = cos(v)
          val List(cId) = node.inputs: @unchecked
          result((node.id, cId)) = num.cos(graph.nodes(cId).value)

        case NodeKind.Op(OpType.Cos) =>
          // d/dv cos(v) = -sin(v)
          val List(cId) = node.inputs: @unchecked
          result((node.id, cId)) = num.negate(num.sin(graph.nodes(cId).value))

        case NodeKind.Op(OpType.Abs) =>
          // d/dv |v| = sign(v)
          val List(cId) = node.inputs: @unchecked
          result((node.id, cId)) = num.signum(graph.nodes(cId).value)

        case NodeKind.Op(OpType.Step) | NodeKind.Op(OpType.Signum) =>
          val List(cId) = node.inputs: @unchecked
          result((node.id, cId)) = num.fromInt(0)

        case _ => () // leaf nodes

    result.toMap

  object syntax:
    extension [A: NumberLike](expr: Value[A])
      def toGraph: TracedGraph[A] = Trace.forward(expr)

    extension [A: NumberLike](graph: TracedGraph[A])
      def gradients: Map[Int, A] = Trace.backward(graph)
      def varGradients: Map[String, A] =
        Trace.varGrads(graph, Trace.backward(graph))
