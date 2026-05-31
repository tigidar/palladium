package palladium.ein

object EinMermaid:

  private def dimStr(dims: List[Dim]): String =
    if dims.isEmpty then "scalar"
    else dims.map(d => s"${d.size}").mkString("[", " x ", "]")

  private def dimLabel(dims: List[Dim]): String =
    if dims.isEmpty then "scalar"
    else dims.map(d => s"${d.name}=${d.size}").mkString("[", ", ", "]")

  def renderOps(graph: EinTracedGraph): String =
    val sb = StringBuilder()
    sb.append("flowchart TD\n")

    // Style classes
    sb.append("    classDef param fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000\n")
    sb.append("    classDef input fill:#bbdefb,stroke:#1565c0,stroke-width:2px,color:#000\n")
    sb.append("    classDef op fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000\n")
    sb.append("    classDef activation fill:#e0f2f1,stroke:#00897b,stroke-width:2px,color:#000\n")
    sb.append("    classDef fill_node fill:#fce4ec,stroke:#c62828,stroke-width:2px,color:#000\n")

    for node <- graph.nodes do
      val shape = dimStr(node.outputDims)
      val (nodeShape, cls) = node.kind match
        case EinNodeKind.Parameter(name) =>
          (s"""["${name}<br/>${shape}"]""", "param")
        case EinNodeKind.Input(name) =>
          (s"""["${name}<br/>${shape}"]""", "input")
        case EinNodeKind.Fill =>
          (s"""("fill<br/>${shape}")""", "fill_node")
        case EinNodeKind.Contract =>
          (s"""(("contract<br/>${shape}"))""", "op")
        case EinNodeKind.ElemAdd =>
          (s"""(("+<br/>${shape}"))""", "op")
        case EinNodeKind.ElemSub =>
          (s"""(("-<br/>${shape}"))""", "op")
        case EinNodeKind.ElemMul =>
          (s"""((".*<br/>${shape}"))""", "op")
        case EinNodeKind.Activate(f) =>
          val name = activationName(f)
          (s"""(["${name}<br/>${shape}"])""", "activation")
        case EinNodeKind.ActivateDeriv(f) =>
          val name = activationName(f) + "'"
          (s"""(["${name}<br/>${shape}"])""", "activation")
        case EinNodeKind.ReduceSum(over) =>
          (s"""(("sum(${over.mkString(",")})<br/>${shape}"))""", "op")
        case EinNodeKind.Broadcast =>
          (s"""(("broadcast<br/>${shape}"))""", "op")
        case EinNodeKind.Transpose =>
          (s"""(("transpose<br/>${shape}"))""", "op")
        case EinNodeKind.Softmax(overDim) =>
          (s"""(["softmax($overDim)<br/>${shape}"])""", "activation")
        case EinNodeKind.LogSoftmax(overDim) =>
          (s"""(["log_softmax($overDim)<br/>${shape}"])""", "activation")
        case EinNodeKind.LayerNorm(overDims, _) =>
          (s"""(["layernorm(${overDims.mkString(",")})<br/>${shape}"])""", "activation")
        case EinNodeKind.Reshape(targetDims) =>
          val target = targetDims.map(d => s"${d.name}=${d.size}").mkString("[", ", ", "]")
          (s"""(("reshape<br/>${target}"))""", "op")
        case EinNodeKind.Slice(dim, from, to) =>
          (s"""(("slice($dim,$from:$to)<br/>${shape}"))""", "op")
        case EinNodeKind.Gather(lookupDim, _) =>
          (s"""(("gather($lookupDim)<br/>${shape}"))""", "op")
        case EinNodeKind.Scatter(lookupDim, _) =>
          (s"""(("scatter($lookupDim)<br/>${shape}"))""", "op")

      sb.append(s"    n${node.id}${nodeShape}:::${cls}\n")

    // Edges with dimension info
    for node <- graph.nodes; childId <- node.inputs do
      val childDims = dimStr(graph.nodes(childId).outputDims)
      sb.append(s"""    n${childId} -->|"${childDims}"| n${node.id}\n""")

    sb.toString

  def renderArch(arch: NetworkArch): String =
    val sb = StringBuilder()
    sb.append("flowchart TD\n")

    // Style classes
    sb.append("    classDef input fill:#bbdefb,stroke:#1565c0,stroke-width:2px,color:#000\n")
    sb.append("    classDef dense fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000\n")
    sb.append("    classDef activation fill:#e0f2f1,stroke:#00897b,stroke-width:2px,color:#000\n")
    sb.append("    classDef output fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px,color:#000\n")

    for layer <- arch.layers do
      val cls = layer.layerType match
        case LayerType.Dense(_, _, _) => "dense"
        case LayerType.ActivationOnly(_) => "activation"
        case LayerType.Custom(_) => "dense"

      val paramInfo = layer.params.map { p =>
        s"${p.name}: ${dimStr(p.dims)}"
      }.mkString("<br/>")

      val paramCount = layer.params.map(_.elementCount).sum

      val details = layer.layerType match
        case LayerType.Dense(inDim, outDim, _) =>
          val base = s"${layer.label}<br/>${dimStr(List(inDim))} → ${dimStr(List(outDim))}"
          val withParams = if paramInfo.nonEmpty then s"$base<br/>$paramInfo" else base
          s"$withParams<br/>params: $paramCount"
        case LayerType.ActivationOnly(f) =>
          layer.label
        case LayerType.Custom(label) =>
          label

      val isLast = !arch.connections.exists(_._1 == layer.id)
      val actualCls = (isLast, layer.layerType) match
        case (true, _: LayerType.Dense) => "output"
        case _                          => cls

      sb.append(s"""    layer${layer.id}["${details}"]:::${actualCls}\n""")

    // Connections with shape transformations
    for (from, to) <- arch.connections do
      val fromDims = dimStr(arch.layers(from).outputDims)
      val toDims = dimStr(arch.layers(to).inputDims)
      sb.append(s"""    layer${from} -->|"${fromDims}"| layer${to}\n""")

    sb.toString

  private def activationName(f: Activation): String = f match
    case Activation.ReLU    => "ReLU"
    case Activation.Sigmoid => "Sigmoid"
    case Activation.Tanh    => "Tanh"
    case Activation.GELU    => "GELU"
    case Activation.Swish   => "Swish"

  object syntax:
    extension [A](graph: EinTracedGraph)
      def toOpsMermaid: String = EinMermaid.renderOps(graph)

    extension (arch: NetworkArch)
      def toArchMermaid: String = EinMermaid.renderArch(arch)
