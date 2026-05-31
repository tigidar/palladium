package frontend

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import org.scalajs.dom
import palladium.*
import palladium.nn.MLP
import palladium.ein.*

/** Scala.js exports for Vite integration.
  *
  * These functions are exported to JavaScript and can be called from the Vite
  * frontend to interact with the palladium library directly in the browser.
  */
@JSExportTopLevel("PaladiumFrontend")
object Main:

  // Import the given NumberLike[Double] from the companion object

  /** Evaluate a simple expression with the given variable values.
    *
    * @param expr
    *   Expression string (e.g., "x * y + z")
    * @param variables
    *   JavaScript object mapping variable names to values
    * @return
    *   The computed result
    */
  @JSExport
  def evaluateExpression(expr: String, variables: js.Dictionary[Double]): Double =
    val (_, exprBody) = splitAssignment(expr)
    val vars = variables.toMap
    val parsed = parseExpression(exprBody, vars)
    parsed.eval

  /** Get the symbolic gradient expressions for all variables in an expression.
    *
    * @param expr
    *   Expression string (e.g., "x^2 + y^2" or "d = x^2 + y^2")
    * @return
    *   JavaScript object mapping variable names to gradient expression strings
    */
  @JSExport
  def getSymbolicGradients(expr: String): js.Dictionary[String] =
    val (_, exprBody) = splitAssignment(expr)
    val varPattern = "[a-zA-Z_][a-zA-Z0-9_]*".r
    val reserved = Set("log", "exp", "tanh", "sigmoid", "relu", "sin", "cos", "abs")
    val varNames = varPattern.findAllIn(exprBody).toSet -- reserved
    val vars = varNames.map(name => name -> 0.0).toMap

    val parsed = parseExpression(exprBody, vars)
    val grads = SymbolicGrad.backward(parsed)

    js.Dictionary(grads.map { case (k, v) => k -> valueToString(v) }.toSeq*)

  /** Get numerical gradients for all variables.
    *
    * @param expr
    *   Expression string
    * @param variables
    *   JavaScript object mapping variable names to values
    * @return
    *   JavaScript object with 'value' and 'gradients' properties
    */
  @JSExport
  def getGradients(expr: String, variables: js.Dictionary[Double]): js.Dynamic =
    val (_, exprBody) = splitAssignment(expr)
    val vars = variables.toMap
    val parsed = parseExpression(exprBody, vars)
    val value = parsed.eval
    val grads = Grad.backward(parsed)

    js.Dynamic.literal(
      value = value,
      gradients = js.Dictionary(grads.toSeq*)
    )

  /** Convert an expression to a Mermaid graph using TracedGraph with gradient annotations.
    *
    * @param expr
    *   Expression string
    * @param variables
    *   JavaScript object mapping variable names to values
    * @return
    *   Mermaid graph definition string with values and gradients
    */
  @JSExport
  def toMermaidGraph(expr: String, variables: js.Dictionary[Double]): String =
    val (resultName, exprBody) = splitAssignment(expr)
    val vars = variables.toMap
    val parsed = parseExpression(exprBody, vars)
    val graph = Trace.forward(parsed)
    val grads = Trace.backward(graph)
    val localDerivs = Trace.localDerivatives(graph)
    Mermaid.render(graph, grads, resultName, localDerivs)

  /** Convert an expression to a D3-compatible JSON graph using TracedGraph.
    *
    * @param expr
    *   Expression string
    * @param variables
    *   JavaScript object mapping variable names to values
    * @return
    *   JSON string with nodes (including values and gradients) and links arrays
    */
  @JSExport
  def toD3Graph(expr: String, variables: js.Dictionary[Double]): String =
    val (resultName, exprBody) = splitAssignment(expr)
    val vars = variables.toMap
    val parsed = parseExpression(exprBody, vars)
    val graph = Trace.forward(parsed)
    val grads = Trace.backward(graph)
    val localDerivs = Trace.localDerivatives(graph)
    tracedGraphToD3Json(graph, grads, resultName, localDerivs)

  @JSExport
  def listNetworks(): js.Array[String] =
    js.Array(NetworkExamples.names*)

  @JSExport
  def getNetworkArchMermaid(name: String): String =
    val expr = NetworkExamples.all.getOrElse(name, throw Exception(s"Unknown network: $name"))
    val graph = EinTrace.forward(expr)
    val arch = EinArch.detect(graph)
    EinMermaid.renderArch(arch)

  @JSExport
  def getNetworkOpsMermaid(name: String): String =
    val expr = NetworkExamples.all.getOrElse(name, throw Exception(s"Unknown network: $name"))
    val graph = EinTrace.forward(expr)
    EinMermaid.renderOps(graph)

  @JSExport
  def buildFromTopology(topologyStr: String, view: String): String =
    val layers = parseTopologyStr(topologyStr)
    val expr = NetworkExamples.fromTopology(layers)
    val graph = EinTrace.forward(expr)
    if view == "arch" then
      val arch = EinArch.detect(graph)
      EinMermaid.renderArch(arch)
    else
      EinMermaid.renderOps(graph)

  @JSExport
  def getTopologySourceCode(topologyStr: String): String =
    val layers = parseTopologyStr(topologyStr)
    NetworkExamples.toScalaSource(layers)

  @JSExport
  def getNetworkSourceCode(name: String): String =
    NetworkExamples.sourceFor(name)

  private def parseTopologyStr(s: String): List[(Int, String)] =
    s.split(",").toList.zipWithIndex.map { case (segment, idx) =>
      val trimmed = segment.trim
      val parts = trimmed.split(":").map(_.trim)
      val size = parts(0).toInt
      val activation =
        if parts.length > 1 then parts(1)
        else if idx == 0 then ""                // input layer: no activation
        else ""                                 // default handled by caller
      (size, activation)
    }

  @JSExport
  def getNetworkArchD3(name: String): String =
    val expr = NetworkExamples.all.getOrElse(name, throw Exception(s"Unknown network: $name"))
    val graph = EinTrace.forward(expr)
    val arch = EinArch.detect(graph)
    archToD3Json(arch)

  private def archToD3Json(arch: NetworkArch): String =
    val nodes = arch.layers.map { layer =>
      val paramInfo = layer.params.map(p => s"${p.name}: ${p.dims.map(_.size).mkString("x")}").mkString(", ")
      val paramCount = layer.params.map(_.elementCount).sum
      s"""{"id":${layer.id},"label":"${escapeJson(layer.label)}","type":"${layerTypeStr(layer.layerType)}","params":"${escapeJson(paramInfo)}","paramCount":$paramCount,"inputDims":"${escapeJson(dimsStr(layer.inputDims))}","outputDims":"${escapeJson(dimsStr(layer.outputDims))}"}"""
    }
    val links = arch.connections.map { case (from, to) =>
      s"""{"source":$from,"target":$to}"""
    }
    s"""{"nodes":${nodes.mkString("[", ",", "]")},"links":${links.mkString("[", ",", "]")}}"""

  private def layerTypeStr(lt: LayerType): String = lt match
    case LayerType.Dense(_, _, _)     => "dense"
    case LayerType.ActivationOnly(_)  => "activation"
    case LayerType.Custom(_)          => "custom"

  private def dimsStr(dims: List[Dim]): String =
    if dims.isEmpty then "scalar"
    else dims.map(d => s"${d.size}").mkString("[", " x ", "]")

  /** Split "d = expr" into (Some("d"), "expr"), or (None, original) if no assignment. */
  private def splitAssignment(expr: String): (Option[String], String) =
    val eqIdx = expr.indexOf('=')
    if eqIdx > 0 then
      val lhs = expr.substring(0, eqIdx).trim
      if lhs.matches("[a-zA-Z_][a-zA-Z0-9_]*") then (Some(lhs), expr.substring(eqIdx + 1).trim)
      else (None, expr)
    else (None, expr)

  // Simple expression parser
  private def parseExpression(expr: String, vars: Map[String, Double]): Value[Double] =
    val tokens = tokenize(expr)
    val (result, _) = parseAddSub(tokens, vars)
    result

  private def tokenize(expr: String): List[String] =
    expr
      .replaceAll("\\s+", "")
      .replaceAll("/~", "\u0000FASTDIV\u0000")
      .replaceAll("([+\\-*/()^])", " $1 ")
      .replaceAll("\u0000FASTDIV\u0000", " /~ ")
      .split("\\s+")
      .filter(_.nonEmpty)
      .toList

  private def parseAddSub(tokens: List[String], vars: Map[String, Double]): (Value[Double], List[String]) =
    var (left, remaining) = parseMulDiv(tokens, vars)
    var current = remaining
    while current.nonEmpty && (current.head == "+" || current.head == "-") do
      val op = current.head
      val (right, rest) = parseMulDiv(current.tail, vars)
      left = if op == "+" then left + right else left - right
      current = rest
    (left, current)

  private def parseMulDiv(tokens: List[String], vars: Map[String, Double]): (Value[Double], List[String]) =
    var (left, remaining) = parsePow(tokens, vars)
    var current = remaining
    while current.nonEmpty && Set("*", "/", "/~").contains(current.head) do
      val op = current.head
      val (right, rest) = parsePow(current.tail, vars)
      left = op match
        case "*"  => left * right
        case "/"  => left / right
        case "/~" => left /~ right
      current = rest
    (left, current)

  private def parsePow(tokens: List[String], vars: Map[String, Double]): (Value[Double], List[String]) =
    val (base, remaining) = parseUnary(tokens, vars)
    if remaining.nonEmpty && remaining.head == "^" then
      val (exp, rest) = parsePow(remaining.tail, vars)
      (base ~^ exp, rest)
    else
      (base, remaining)

  private def parseUnary(tokens: List[String], vars: Map[String, Double]): (Value[Double], List[String]) =
    tokens match
      case "-" :: rest =>
        val (value, remaining) = parseUnary(rest, vars)
        (-value, remaining)
      case _ =>
        parsePrimary(tokens, vars)

  private def parsePrimary(tokens: List[String], vars: Map[String, Double]): (Value[Double], List[String]) =
    tokens match
      case "(" :: rest =>
        val (expr, afterExpr) = parseAddSub(rest, vars)
        afterExpr match
          case ")" :: remaining => (expr, remaining)
          case _                => throw Exception("Missing closing parenthesis")
      case "log" :: "(" :: rest =>
        val (expr, afterExpr) = parseAddSub(rest, vars)
        afterExpr match
          case ")" :: remaining => (expr.log, remaining)
          case _                => throw Exception("Missing closing parenthesis for log")
      case "exp" :: "(" :: rest =>
        val (expr, afterExpr) = parseAddSub(rest, vars)
        afterExpr match
          case ")" :: remaining => (expr.exp, remaining)
          case _                => throw Exception("Missing closing parenthesis for exp")
      case "tanh" :: "(" :: rest =>
        val (expr, afterExpr) = parseAddSub(rest, vars)
        afterExpr match
          case ")" :: remaining => (expr.tanh, remaining)
          case _                => throw Exception("Missing closing parenthesis for tanh")
      case "sigmoid" :: "(" :: rest =>
        val (expr, afterExpr) = parseAddSub(rest, vars)
        afterExpr match
          case ")" :: remaining => (expr.sigmoid, remaining)
          case _                => throw Exception("Missing closing parenthesis for sigmoid")
      case "relu" :: "(" :: rest =>
        val (expr, afterExpr) = parseAddSub(rest, vars)
        afterExpr match
          case ")" :: remaining => (expr.relu, remaining)
          case _                => throw Exception("Missing closing parenthesis for relu")
      case "sin" :: "(" :: rest =>
        val (expr, afterExpr) = parseAddSub(rest, vars)
        afterExpr match
          case ")" :: remaining => (expr.sin, remaining)
          case _                => throw Exception("Missing closing parenthesis for sin")
      case "cos" :: "(" :: rest =>
        val (expr, afterExpr) = parseAddSub(rest, vars)
        afterExpr match
          case ")" :: remaining => (expr.cos, remaining)
          case _                => throw Exception("Missing closing parenthesis for cos")
      case "abs" :: "(" :: rest =>
        val (expr, afterExpr) = parseAddSub(rest, vars)
        afterExpr match
          case ")" :: remaining => (expr.abs, remaining)
          case _                => throw Exception("Missing closing parenthesis for abs")
      case token :: rest =>
        val value = token.toDoubleOption match
          case Some(d) => Value.Lit(d)
          case None =>
            vars.get(token) match
              case Some(v) => Value.Var(token, v)
              case None    => throw Exception(s"Unknown variable: $token")
        (value, rest)
      case Nil =>
        throw Exception("Unexpected end of expression")

  private def valueToString[A](v: Value[A]): String =
    import Value.*
    v match
      case Var(id, _)     => id
      case Lit(data)      => data.toString
      case Const(n)       => n.toString
      case Add(l, r)      => s"(${valueToString(l)} + ${valueToString(r)})"
      case Sub(l, r)      => s"(${valueToString(l)} - ${valueToString(r)})"
      case Mul(l, r)      => s"(${valueToString(l)} * ${valueToString(r)})"
      case Div(l, r)      => s"(${valueToString(l)} / ${valueToString(r)})"
      case FastDiv(l, r)  => s"(${valueToString(l)} /~ ${valueToString(r)})"
      case Pow(base, exp) => s"(${valueToString(base)} ^ ${valueToString(exp)})"
      case Neg(value)     => s"(-${valueToString(value)})"
      case Log(value)     => s"log(${valueToString(value)})"
      case Exp(value)     => s"exp(${valueToString(value)})"
      case Tanh(value)    => s"tanh(${valueToString(value)})"
      case Sigmoid(value) => s"sigmoid(${valueToString(value)})"
      case Relu(value)    => s"relu(${valueToString(value)})"
      case Sin(value)     => s"sin(${valueToString(value)})"
      case Cos(value)     => s"cos(${valueToString(value)})"
      case Abs(value)     => s"abs(${valueToString(value)})"
      case Step(value)    => s"step(${valueToString(value)})"
      case Signum(value)  => s"sgn(${valueToString(value)})"

  private def tracedGraphToD3Json(
      graph: TracedGraph[Double],
      grads: Map[Int, Double],
      resultName: Option[String] = None,
      localDerivs: Map[(Int, Int), Double] = Map.empty
  ): String =
    val graphNodes = graph.nodes.map { node =>
      val nodeType = node.kind match
        case NodeKind.Variable(_)    => "variable"
        case NodeKind.Literal        => "literal"
        case NodeKind.Constant       => "constant"
        case NodeKind.Op(OpType.Neg) => "unary"
        case NodeKind.Op(OpType.Log | OpType.Exp | OpType.Tanh | OpType.Sigmoid |
                         OpType.Relu | OpType.Sin | OpType.Cos | OpType.Abs | OpType.Step | OpType.Signum) => "function"
        case NodeKind.Op(_)          => "operation"
      val grad = grads.getOrElse(node.id, 0.0)
      s"""{"id":${node.id},"label":"${escapeJson(node.label)}","type":"$nodeType","value":${node.value},"gradient":$grad}"""
    }

    val graphLinks = for
      node <- graph.nodes
      (inputId, idx) <- node.inputs.zipWithIndex
    yield
      val ld = localDerivs.getOrElse((node.id, inputId), 0.0)
      val parentGrad = grads.getOrElse(node.id, 0.0)
      val gradContrib = parentGrad * ld
      val inputLabel = escapeJson(graph.nodes(inputId).label)
      s"""{"source":$inputId,"target":${node.id},"localDeriv":$ld,"inputLabel":"$inputLabel","gradContrib":$gradContrib,"linkIndex":$idx}"""

    // Append result node if named
    val (allNodes, allLinks) = resultName match
      case Some(name) =>
        val resultId = graph.nodes.size
        val rootValue = graph.nodes(graph.rootId).value
        val rootLabel = escapeJson(graph.nodes(graph.rootId).label)
        val resultNode = s"""{"id":$resultId,"label":"${escapeJson(name)}","type":"result","value":$rootValue,"gradient":1.0}"""
        val resultLink = s"""{"source":${graph.rootId},"target":$resultId,"localDeriv":1.0,"inputLabel":"$rootLabel","gradContrib":1.0,"linkIndex":0}"""
        (graphNodes :+ resultNode, graphLinks :+ resultLink)
      case None =>
        (graphNodes, graphLinks)

    s"""{"nodes":${allNodes.mkString("[", ",", "]")},"links":${allLinks.mkString("[", ",", "]")}}"""

  // ============================================
  // Neural Network (MLP) exports
  // ============================================

  /** Initialize an MLP with the given topology and return its parameters as JSON.
    *
    * @param topology
    *   Array of layer sizes, e.g. [2, 3, 1]
    * @param seed
    *   Random seed for weight initialization
    * @return
    *   JSON string with params map and layer info
    */
  @JSExport
  def initMLP(topology: js.Array[Int], seed: Double): String =
    val layers = topology.toVector
    val mlp = MLP.initFromVector(layers, seed.toLong)
    val paramsJson = mlp.params.toSeq.sortBy(_._1).map { case (k, v) =>
      s""""$k":$v"""
    }.mkString("{", ",", "}")
    s"""{"params":$paramsJson,"layers":${layers.mkString("[", ",", "]")},"paramCount":${mlp.paramCount}}"""

  /** Run MLP forward pass and return output values.
    *
    * @param topology
    *   Array of layer sizes
    * @param params
    *   Parameter values as name->value dictionary
    * @param inputs
    *   Input values
    * @return
    *   JSON string with outputs array
    */
  @JSExport
  def mlpForward(topology: js.Array[Int], params: js.Dictionary[Double], inputs: js.Array[Double]): String =
    val layers = topology.toVector
    val mlp = MLP.fromVector(layers, params.toMap)
    val inputValues = inputs.toVector.map(Value.Lit(_))
    val outputs = mlp.forward(inputValues)
    val outputValues = outputs.map(_.eval)
    outputValues.mkString("[", ",", "]")

  /** Compute gradients for all MLP parameters given inputs and a target (MSE loss).
    *
    * @param topology
    *   Array of layer sizes
    * @param params
    *   Parameter values
    * @param inputs
    *   Input values
    * @param targets
    *   Target output values
    * @return
    *   JSON string with loss value and gradients map
    */
  @JSExport
  def mlpGradients(
      topology: js.Array[Int],
      params: js.Dictionary[Double],
      inputs: js.Array[Double],
      targets: js.Array[Double]
  ): String =
    val layers = topology.toVector
    val mlp = MLP.fromVector(layers, params.toMap)
    val inputValues = inputs.toVector.map(Value.Lit(_))
    val targetValues = targets.toVector.map(Value.Lit(_))
    val outputs = mlp.forward(inputValues)
    val loss = Loss.mse(outputs, targetValues)
    val lossValue = loss.eval
    val grads = Grad.backward(loss)
    val gradsJson = grads.toSeq.sortBy(_._1).map { case (k, v) =>
      s""""$k":$v"""
    }.mkString("{", ",", "}")
    s"""{"loss":$lossValue,"gradients":$gradsJson}"""

  /** Run one SGD training step: forward, compute loss+gradients, update params.
    *
    * @param topology
    *   Array of layer sizes
    * @param params
    *   Current parameter values
    * @param inputs
    *   Input values
    * @param targets
    *   Target output values
    * @param learningRate
    *   SGD learning rate
    * @return
    *   JSON string with updated params, loss, and outputs
    */
  @JSExport
  def mlpTrainStep(
      topology: js.Array[Int],
      params: js.Dictionary[Double],
      inputs: js.Array[Double],
      targets: js.Array[Double],
      learningRate: Double
  ): String =
    val layers = topology.toVector
    val mlp = MLP.fromVector(layers, params.toMap)
    val inputValues = inputs.toVector.map(Value.Lit(_))
    val targetValues = targets.toVector.map(Value.Lit(_))
    val outputs = mlp.forward(inputValues)
    val loss = Loss.mse(outputs, targetValues)
    val lossValue = loss.eval
    val grads = Grad.backward(loss)
    val sgd = Optimizer.SGD[Double](learningRate)
    val updatedParams = sgd.step(mlp.params, grads)
    val paramsJson = updatedParams.toSeq.sortBy(_._1).map { case (k, v) =>
      s""""$k":$v"""
    }.mkString("{", ",", "}")
    val outputValues = outputs.map(_.eval)
    s"""{"params":$paramsJson,"loss":$lossValue,"outputs":${outputValues.mkString("[", ",", "]")}}"""

  /** Generate a D3-compatible graph for an MLP forward pass.
    *
    * @param topology
    *   Array of layer sizes
    * @param params
    *   Parameter values
    * @param inputs
    *   Input values
    * @return
    *   JSON string with nodes and links for D3 visualization
    */
  @JSExport
  def mlpToD3Graph(
      topology: js.Array[Int],
      params: js.Dictionary[Double],
      inputs: js.Array[Double]
  ): String =
    val layers = topology.toVector
    val mlp = MLP.fromVector(layers, params.toMap)
    val inputValues = inputs.toVector.zipWithIndex.map { case (v, i) => Value.Var(s"x$i", v) }
    val outputs = mlp.forward(inputValues)
    val output = outputs.head
    val graph = Trace.forward(output)
    val grads = Trace.backward(graph)
    val localDerivs = Trace.localDerivatives(graph)
    tracedGraphToD3Json(graph, grads, Some("output"), localDerivs)

  // ============================================
  // Block Composition DSL exports
  // ============================================

  /** Build a network from a Block DSL specification and return Mermaid diagram.
    *
    * @param spec
    *   Block spec string, e.g. "784, dense:128:relu * 3, linear:10"
    * @param view
    *   "arch" or "ops"
    * @return
    *   Mermaid graph definition string
    */
  @JSExport
  def buildBlockNetwork(spec: String, view: String): String =
    val (input, block) = parseBlockSpec(spec)
    val expr = block.materialize(input)
    val graph = EinTrace.forward(expr)
    if view == "arch" then
      val arch = EinArch.detect(graph)
      EinMermaid.renderArch(arch)
    else
      EinMermaid.renderOps(graph)

  /** Build a network from a Block DSL specification and return Scala source code. */
  @JSExport
  def getBlockSourceCode(spec: String): String =
    val (input, block) = parseBlockSpec(spec)
    blockSpecToSource(spec, input, block)

  /** Parse a block spec like "784, dense:128:relu * 3, linear:10".
    *
    * Format: inputSize, block1, block2, ...
    * Block formats:
    *   - dense:outSize:activation   (e.g. dense:128:relu)
    *   - dense:outSize:activation * N  (repeat N times)
    *   - linear:outSize             (no activation)
    *   - residual(block >> block)    (skip connection — future)
    */
  private def parseBlockSpec(spec: String): (Ein[Double], Block[Double]) =
    // Split on comma, but respect parentheses
    val segments = splitTopLevel(spec, ',').map(_.trim).filter(_.nonEmpty)
    require(segments.nonEmpty, "Block spec must have at least an input size")

    // First segment is the input size
    val inputSize = segments.head.toInt
    val inputDim = Dim("inp", inputSize)
    val input = Ein.Input[Double]("x", List(inputDim))

    // Remaining segments are blocks
    val blocks = segments.tail.map(parseBlockSegment)
    require(blocks.nonEmpty, "Block spec must have at least one block")

    val combined = blocks.reduceLeft(_ >> _)
    (input, combined)

  private def parseBlockSegment(seg: String): Block[Double] =
    // Check for " * N" repeat suffix
    val repeatPattern = """^(.+?)\s*\*\s*(\d+)$""".r
    // Check for "residual(...)"
    val residualPattern = """^residual\((.+)\)$""".r

    seg match
      case repeatPattern(inner, n) =>
        parseBlockSegment(inner.trim) * n.toInt
      case residualPattern(inner) =>
        val innerBlocks = splitTopLevel(inner, '>').map(_.trim.stripPrefix(">").trim).filter(_.nonEmpty)
        val combined = innerBlocks.map(parseSingleBlock).reduceLeft(_ >> _)
        combined.residual
      case _ =>
        parseSingleBlock(seg)

  private def parseSingleBlock(s: String): Block[Double] =
    val parts = s.split(":").map(_.trim)
    parts(0).toLowerCase match
      case "dense" =>
        require(parts.length >= 3, s"dense block needs size and activation: $s")
        val size = parts(1).toInt
        val act = parseActivation(parts(2))
        Block.dense[Double](size, act)
      case "linear" =>
        require(parts.length >= 2, s"linear block needs size: $s")
        val size = parts(1).toInt
        Block.linear[Double](size)
      case other =>
        throw IllegalArgumentException(s"Unknown block type: $other")

  private def parseActivation(name: String): Activation =
    name.toLowerCase.trim match
      case "relu"    => Activation.ReLU
      case "sigmoid" => Activation.Sigmoid
      case "tanh"    => Activation.Tanh
      case other     => throw IllegalArgumentException(s"Unknown activation: $other")

  private def splitTopLevel(s: String, delimiter: Char): List[String] =
    val result = scala.collection.mutable.ListBuffer[String]()
    val current = new StringBuilder
    var depth = 0
    for c <- s do
      if c == '(' then depth += 1
      else if c == ')' then depth -= 1
      if c == delimiter && depth == 0 then
        result += current.toString
        current.clear()
      else
        current += c
    if current.nonEmpty then result += current.toString
    result.toList

  private def blockSpecToSource(spec: String, input: Ein[Double], block: Block[Double]): String =
    val segments = splitTopLevel(spec, ',').map(_.trim).filter(_.nonEmpty)
    val inputSize = segments.head
    val blockSegments = segments.tail

    val sb = new StringBuilder
    sb.append("import palladium.ein.*\n")
    sb.append("import Block.*\n\n")
    sb.append(s"""val inp = Ein.Input[Double]("x", List(Dim("inp", $inputSize)))\n\n""")

    val blockExprs = blockSegments.map(segmentToScala)
    sb.append(s"val net = (inp >> ${blockExprs.mkString(" >> ")}).materialize\n")
    sb.toString

  private def segmentToScala(seg: String): String =
    val repeatPattern = """^(.+?)\s*\*\s*(\d+)$""".r
    val residualPattern = """^residual\((.+)\)$""".r
    seg match
      case repeatPattern(inner, n) =>
        s"${segmentToScala(inner.trim)} * $n"
      case residualPattern(inner) =>
        val parts = splitTopLevel(inner, '>').map(_.trim.stripPrefix(">").trim).filter(_.nonEmpty)
        val combined = parts.map(singleBlockToScala).mkString(" >> ")
        s"($combined).residual"
      case _ =>
        singleBlockToScala(seg)

  private def singleBlockToScala(s: String): String =
    val parts = s.split(":").map(_.trim)
    parts(0).toLowerCase match
      case "dense"  =>
        val actName = parts(2).toLowerCase match
          case "relu"    => "ReLU"
          case "sigmoid" => "Sigmoid"
          case "tanh"    => "Tanh"
          case other     => other.capitalize
        s"Block.dense[Double](${parts(1)}, Activation.$actName)"
      case "linear" => s"Block.linear[Double](${parts(1)})"
      case other    => s"/* unknown: $other */"

  private def escapeJson(s: String): String =
    s.replace("\\", "\\\\").replace("\"", "\\\"")
