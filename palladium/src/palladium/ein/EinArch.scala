package palladium.ein

case class ArchParam(name: String, dims: List[Dim], elementCount: Int)

enum LayerType:
  case Dense(inDim: Dim, outDim: Dim, activation: Option[Activation])
  case ActivationOnly(f: Activation)
  case Custom(label: String)

case class ArchLayer(
    id: Int,
    layerType: LayerType,
    label: String,
    inputDims: List[Dim],
    outputDims: List[Dim],
    params: List[ArchParam],
    nodeIds: Set[Int]
)

case class NetworkArch(
    layers: Vector[ArchLayer],
    connections: Vector[(Int, Int)],
    totalParams: Int
)

object EinArch:

  def detect(graph: EinTracedGraph): NetworkArch =
    val nodes = graph.nodes
    val consumed = scala.collection.mutable.Set.empty[Int]
    val layers = scala.collection.mutable.ArrayBuffer.empty[ArchLayer]

    // Process nodes from root backward (reverse topological order)
    for node <- nodes.reverseIterator if !consumed.contains(node.id) do
      matchLayer(nodes, node, consumed) match
        case Some(layer) =>
          consumed ++= layer.nodeIds
          layers += layer
        case None => ()

    // Reverse to get input→output order
    val orderedLayers = layers.reverse.toVector
    val reindexed = orderedLayers.zipWithIndex.map { case (l, idx) =>
      l.copy(id = idx)
    }

    // Build connections: layer i → layer j when j's input nodes overlap with i's output node
    val nodeToLayer = scala.collection.mutable.Map.empty[Int, Int]
    for layer <- reindexed; nid <- layer.nodeIds do
      nodeToLayer(nid) = layer.id

    val connections = scala.collection.mutable.Set.empty[(Int, Int)]
    for layer <- reindexed do
      // Find the graph nodes in this layer that take inputs from other layers
      for nid <- layer.nodeIds do
        val node = nodes(nid)
        for inputId <- node.inputs do
          nodeToLayer.get(inputId) match
            case Some(srcLayerId) if srcLayerId != layer.id =>
              connections += ((srcLayerId, layer.id))
            case _ => ()

    val totalParams = reindexed.flatMap(_.params).map(_.elementCount).sum

    NetworkArch(reindexed, connections.toVector.sorted, totalParams)

  private def matchLayer(
      nodes: Vector[EinTracedNode],
      node: EinTracedNode,
      consumed: scala.collection.mutable.Set[Int]
  ): Option[ArchLayer] =
    // Pattern 1: Activate(f, ElemAdd(Contract(Param, _), Param)) — Dense + activation
    // Pattern 2: ElemAdd(Contract(Param, _), Param) — Dense without activation
    // Pattern 3: Activate(f, _) — standalone activation
    node.kind match
      case EinNodeKind.Activate(f) =>
        val List(argId) = node.inputs: @unchecked
        val argNode = nodes(argId)
        if !consumed.contains(argId) then
          matchDenseAdd(nodes, argNode, consumed) match
            case Some((weight, bias, contractNode, addNode, inputDims, outDim, inDim)) =>
              val nodeIds = Set(node.id, addNode.id, contractNode.id, weight.id, bias.id)
              val wParams = ArchParam(weight.label, weight.outputDims, weight.outputDims.map(_.size).product)
              val bParams = ArchParam(bias.label, bias.outputDims, bias.outputDims.map(_.size).product)
              Some(ArchLayer(
                0, LayerType.Dense(inDim, outDim, Some(f)),
                s"Dense + ${activationName(f)}",
                inputDims, node.outputDims,
                List(wParams, bParams), nodeIds
              ))
            case None =>
              // Standalone activation
              Some(ArchLayer(
                0, LayerType.ActivationOnly(f),
                activationName(f),
                argNode.outputDims, node.outputDims,
                Nil, Set(node.id)
              ))
        else
          // Standalone activation (arg already consumed)
          Some(ArchLayer(
            0, LayerType.ActivationOnly(f),
            activationName(f),
            nodes(argId).outputDims, node.outputDims,
            Nil, Set(node.id)
          ))

      case EinNodeKind.ElemAdd =>
        if !consumed.contains(node.id) then
          matchDenseAdd(nodes, node, consumed) match
            case Some((weight, bias, contractNode, addNode, inputDims, outDim, inDim)) =>
              val nodeIds = Set(addNode.id, contractNode.id, weight.id, bias.id)
              val wParams = ArchParam(weight.label, weight.outputDims, weight.outputDims.map(_.size).product)
              val bParams = ArchParam(bias.label, bias.outputDims, bias.outputDims.map(_.size).product)
              Some(ArchLayer(
                0, LayerType.Dense(inDim, outDim, None),
                "Dense (linear)",
                inputDims, node.outputDims,
                List(wParams, bParams), nodeIds
              ))
            case None => None
        else None

      case EinNodeKind.Softmax(overDim) =>
        Some(ArchLayer(
          0, LayerType.Custom(s"Softmax($overDim)"),
          s"Softmax($overDim)",
          node.outputDims, node.outputDims,
          Nil, Set(node.id)
        ))

      case EinNodeKind.LogSoftmax(overDim) =>
        Some(ArchLayer(
          0, LayerType.Custom(s"LogSoftmax($overDim)"),
          s"LogSoftmax($overDim)",
          node.outputDims, node.outputDims,
          Nil, Set(node.id)
        ))

      case EinNodeKind.LayerNorm(overDims, _) =>
        val params = node.inputs.drop(1).flatMap { childId =>
          nodes(childId).kind match
            case EinNodeKind.Parameter(name) =>
              val pNode = nodes(childId)
              Some(ArchParam(name, pNode.outputDims, pNode.outputDims.map(_.size).product))
            case _ => None
        }
        Some(ArchLayer(
          0, LayerType.Custom(s"LayerNorm(${overDims.mkString(",")})"),
          s"LayerNorm(${overDims.mkString(",")})",
          node.outputDims, node.outputDims,
          params, Set(node.id) ++ node.inputs.drop(1).toSet
        ))

      case EinNodeKind.Gather(lookupDim, indexDims) =>
        val tableNodeId = node.inputs.head
        val tableNode = nodes(tableNodeId)
        val params = tableNode.kind match
          case EinNodeKind.Parameter(_) =>
            List(ArchParam(tableNode.label, tableNode.outputDims, tableNode.outputDims.map(_.size).product))
          case _ => Nil
        Some(ArchLayer(
          0, LayerType.Custom(s"Embedding($lookupDim)"),
          s"Embedding($lookupDim)",
          tableNode.outputDims, node.outputDims,
          params, Set(node.id) ++ (if params.nonEmpty then Set(tableNodeId) else Set.empty)
        ))

      case _ => None

  /** Try to match an ElemAdd node as Contract(Param, _) + Param (dense layer pattern).
    * Returns (weightNode, biasNode, contractNode, addNode, inputDims, outDim, inDim) or None.
    */
  private def matchDenseAdd(
      nodes: Vector[EinTracedNode],
      addNode: EinTracedNode,
      consumed: scala.collection.mutable.Set[Int]
  ): Option[(EinTracedNode, EinTracedNode, EinTracedNode, EinTracedNode, List[Dim], Dim, Dim)] =
    addNode.kind match
      case EinNodeKind.ElemAdd =>
        val List(lId, rId) = addNode.inputs: @unchecked
        val left = nodes(lId)
        val right = nodes(rId)

        // Try both orderings: Contract+Param or Param+Contract
        tryDensePattern(nodes, left, right, addNode, consumed)
          .orElse(tryDensePattern(nodes, right, left, addNode, consumed))

      case _ => None

  private def tryDensePattern(
      nodes: Vector[EinTracedNode],
      contractCandidate: EinTracedNode,
      biasCandidate: EinTracedNode,
      addNode: EinTracedNode,
      consumed: scala.collection.mutable.Set[Int]
  ): Option[(EinTracedNode, EinTracedNode, EinTracedNode, EinTracedNode, List[Dim], Dim, Dim)] =
    (contractCandidate.kind, biasCandidate.kind) match
      case (EinNodeKind.Contract, EinNodeKind.Parameter(_))
          if !consumed.contains(contractCandidate.id) && !consumed.contains(biasCandidate.id) =>
        val List(wId, xId) = contractCandidate.inputs: @unchecked
        val wNode = nodes(wId)
        val xNode = nodes(xId)

        // Weight should be a Param with 2 dims, one matching bias
        (wNode.kind, wNode.outputDims) match
          case (EinNodeKind.Parameter(_), wDims) if wDims.size == 2 =>
            val biasDims = biasCandidate.outputDims
            // Output dim is the one in weight that matches bias
            val outDimOpt = wDims.find(d => biasDims.exists(_.name == d.name))
            val inDimOpt = wDims.find(d => !biasDims.exists(_.name == d.name))
            (outDimOpt, inDimOpt) match
              case (Some(outDim), Some(inDim)) =>
                val inputDims = xNode.outputDims
                Some((wNode, biasCandidate, contractCandidate, addNode, inputDims, outDim, inDim))
              case _ => None
          case _ => None

      case _ => None

  private def activationName(f: Activation): String = f match
    case Activation.ReLU    => "ReLU"
    case Activation.Sigmoid => "Sigmoid"
    case Activation.Tanh    => "Tanh"
    case Activation.GELU    => "GELU"
    case Activation.Swish   => "Swish"
