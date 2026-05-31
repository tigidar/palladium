package palladium

object Mermaid:

  /** Render a traced computation graph as Mermaid flowchart text.
    *
    * @param resultName
    *   If provided, adds a labeled output node (e.g. "d") connected from the root.
    */
  def render[A: NumberLike](
      graph: TracedGraph[A],
      grads: Map[Int, A] = Map.empty,
      resultName: Option[String] = None,
      localDerivs: Map[(Int, Int), A] = Map.empty
  ): String =
    val num = summon[NumberLike[A]]
    val sb = StringBuilder()
    sb.append("flowchart TD\n")

    // Style classes
    sb.append("    classDef var fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000\n")
    sb.append("    classDef lit fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000\n")
    sb.append("    classDef const fill:#fce4ec,stroke:#c62828,stroke-width:2px,color:#000\n")
    sb.append("    classDef op fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000\n")
    sb.append("    classDef result fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px,color:#000\n")

    // Nodes — KaTeX math labels with <br/> line breaks
    for node <- graph.nodes do
      val grad = grads.get(node.id)
      val (shape, cls) = node.kind match
        case NodeKind.Variable(name) =>
          val label = grad match
            case Some(g) => s"$$$$${name} = ${node.value},\\; \\nabla = $g$$$$"
            case None    => s"$$$$${name} = ${node.value}$$$$"
          (s"""["$label"]""", "var")

        case NodeKind.Literal =>
          val label = grad match
            case Some(g) => s"$$$$${node.value},\\; \\nabla = $g$$$$"
            case None    => s"$$$$${node.value}$$$$"
          (s"""("$label")""", "lit")

        case NodeKind.Constant =>
          val label = grad match
            case Some(g) => s"$$$$${node.value},\\; \\nabla = $g$$$$"
            case None    => s"$$$$${node.value}$$$$"
          (s"""("$label")""", "const")

        case NodeKind.Op(op) =>
          val opSymbol = op match
            case OpType.Add => "+"
            case OpType.Sub => "-"
            case OpType.Mul => "\\times"
            case OpType.Div => "\\div"
            case OpType.FastDiv => "\\div_{\\!\\sim}"
            case OpType.Pow => "\\wedge"
            case OpType.Neg => "-"
            case OpType.Log => "\\ln"
            case OpType.Exp => "\\exp"
            case OpType.Tanh => "\\tanh"
            case OpType.Sigmoid => "\\sigma"
            case OpType.Relu => "\\text{ReLU}"
            case OpType.Sin => "\\sin"
            case OpType.Cos => "\\cos"
            case OpType.Abs => "|{\\cdot}|"
            case OpType.Step => "\\text{step}"
            case OpType.Signum => "\\text{sgn}"
          val label = grad match
            case Some(g) => s"$$$$$opSymbol$$$$<br/>$$$$= ${node.value},\\; \\nabla = $g$$$$"
            case None    => s"$$$$$opSymbol$$$$<br/>$$$$= ${node.value}$$$$"
          (s"""(("$label"))""", "op")

      sb.append(s"    n${node.id}${shape}:::${cls}\n")

    // Edges with local derivative and gradient contribution labels
    for node <- graph.nodes; childId <- node.inputs do
      (localDerivs.get((node.id, childId)), grads.get(node.id)) match
        case (Some(ld), Some(parentGrad)) =>
          val contrib = num.times(parentGrad, ld)
          sb.append(s"""    n${childId} -->|"$$$$$ld \\mid $contrib$$$$"| n${node.id}\n""")
        case (Some(ld), None) =>
          sb.append(s"""    n${childId} -->|"$$$$$ld$$$$"| n${node.id}\n""")
        case _ =>
          sb.append(s"    n${childId} --> n${node.id}\n")

    // Optional result node
    for name <- resultName do
      val rootValue = graph.nodes(graph.rootId).value
      val rootLabel = graph.nodes(graph.rootId).label
      val resultId = s"result"
      sb.append(s"""    $resultId[["$$$$$name = $rootValue$$$$"]]:::result\n""")
      sb.append(s"""    n${graph.rootId} -->|"$$$$1 \\mid 1$$$$"| $resultId\n""")

    sb.toString

  object syntax:
    extension [A: NumberLike](graph: TracedGraph[A])
      def toMermaid: String = Mermaid.render(graph)
      def toMermaid(grads: Map[Int, A]): String = Mermaid.render(graph, grads)
