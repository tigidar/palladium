package palladium.ein

object NetworkExamples:

  val all: Map[String, Ein[Double]] = Map(
    "twoLayerMlp" -> twoLayerMlp,
    "mnistClassifier" -> mnistClassifier,
    "tinyTransformer" -> tinyTransformer
  )

  def names: List[String] = all.keys.toList.sorted

  /** 2 → 3 (ReLU) → 1 (linear) */
  def twoLayerMlp: Ein[Double] =
    val inp = Dim("inp", 2)
    val hid = Dim("hid", 3)
    val out = Dim("out", 1)

    val w1 = Ein.Param("W1", List(hid, inp), TensorData.zeros[Double](List(hid, inp)))
    val b1 = Ein.Param("b1", List(hid), TensorData.zeros[Double](List(hid)))
    val w2 = Ein.Param("W2", List(out, hid), TensorData.zeros[Double](List(out, hid)))
    val b2 = Ein.Param("b2", List(out), TensorData.zeros[Double](List(out)))
    val x = Ein.Param("x", List(inp), TensorData.zeros[Double](List(inp)))

    val h = Ein.Activate(Activation.ReLU, (w1 * x) + b1)
    (w2 * h) + b2

  /** 784 → 128 (ReLU) → 64 (ReLU) → 10 (linear) */
  def mnistClassifier: Ein[Double] =
    val inp = Dim("inp", 784)
    val h1 = Dim("h1", 128)
    val h2 = Dim("h2", 64)
    val out = Dim("out", 10)

    val w1 = Ein.Param("W1", List(h1, inp), TensorData.zeros[Double](List(h1, inp)))
    val b1 = Ein.Param("b1", List(h1), TensorData.zeros[Double](List(h1)))
    val w2 = Ein.Param("W2", List(h2, h1), TensorData.zeros[Double](List(h2, h1)))
    val b2 = Ein.Param("b2", List(h2), TensorData.zeros[Double](List(h2)))
    val w3 = Ein.Param("W3", List(out, h2), TensorData.zeros[Double](List(out, h2)))
    val b3 = Ein.Param("b3", List(out), TensorData.zeros[Double](List(out)))
    val x = Ein.Param("x", List(inp), TensorData.zeros[Double](List(inp)))

    val hidden1 = Ein.Activate(Activation.ReLU, (w1 * x) + b1)
    val hidden2 = Ein.Activate(Activation.ReLU, (w2 * hidden1) + b2)
    (w3 * hidden2) + b3

  /** Tiny GPT-2-style decoder-only transformer.
    *
    * 2 layers, 2 heads, headDim=4, model=8, ffDim=16, seqLen=8.
    * Input: [seq(8), model(8)]
    * Output: [seq(8), model(8)]
    *
    * Architecture per layer:
    *   LayerNorm → 2-head Attention (causal) → Residual →
    *   LayerNorm → FFN (GELU, hidden=16) → Residual
    */
  def tinyTransformer: Ein[Double] =
    val seq = Dim("seq", 8)
    val model = Dim("model", 8)
    val input = Ein.Input[Double]("x", List(seq, model))

    val transformer = Block.transformer[Double](
      nLayers = 2,
      nHeads = 2,
      headDim = 4,
      ffDim = 16,
      seqLen = 8
    )
    transformer.materialize(input)

  /** Build a network from a topology specification.
    *
    * @param layers
    *   List of (size, activation) pairs. First entry is the input layer (activation ignored).
    *   Activation: "relu", "sigmoid", "tanh", or "linear"/"none".
    */
  def fromTopology(layers: List[(Int, String)]): Ein[Double] =
    require(layers.length >= 2, "Need at least 2 layers (input + output)")

    val dims = layers.zipWithIndex.map { case ((size, _), idx) =>
      val name =
        if idx == 0 then "inp"
        else if idx == layers.length - 1 then "out"
        else s"h$idx"
      Dim(name, size)
    }

    val x = Ein.Param("x", List(dims.head), TensorData.zeros[Double](List(dims.head)))

    var current: Ein[Double] = x
    for i <- 1 until layers.length do
      val prevDim = dims(i - 1)
      val curDim = dims(i)
      val (_, act) = layers(i)

      val w = Ein.Param(s"W$i", List(curDim, prevDim), TensorData.zeros[Double](List(curDim, prevDim)))
      val b = Ein.Param(s"b$i", List(curDim), TensorData.zeros[Double](List(curDim)))

      val linear = (w * current) + b
      current = activationFor(act) match
        case Some(f) => Ein.Activate(f, linear)
        case None    => linear

    current

  private def activationFor(name: String): Option[Activation] =
    name.toLowerCase.trim match
      case "relu"              => Some(Activation.ReLU)
      case "sigmoid"           => Some(Activation.Sigmoid)
      case "tanh"              => Some(Activation.Tanh)
      case "linear" | "none" | "" => None
      case other               => throw IllegalArgumentException(s"Unknown activation: $other")

  /** Generate readable Scala source code for a network topology. */
  def toScalaSource(layers: List[(Int, String)]): String =
    require(layers.length >= 2, "Need at least 2 layers (input + output)")

    val sb = new StringBuilder

    // Dim declarations
    val dimNames = layers.zipWithIndex.map { case ((size, _), idx) =>
      val name =
        if idx == 0 then "inp"
        else if idx == layers.length - 1 then "out"
        else s"h$idx"
      (name, size)
    }

    for (name, size) <- dimNames do
      val pad = " " * (4 - name.length).max(0)
      sb.append(s"""val $name$pad = Dim("$name", $size)\n""")

    sb.append("\n")

    // Weight and bias declarations
    for i <- 1 until layers.length do
      val prevName = dimNames(i - 1)._1
      val curName = dimNames(i)._1
      sb.append(s"""val W$i = Ein.Param("W$i", List($curName, $prevName), TensorData.zeros(List($curName, $prevName)))\n""")
      sb.append(s"""val b$i = Ein.Param("b$i", List($curName), TensorData.zeros(List($curName)))\n""")

    sb.append(s"""val x  = Ein.Param("x", List(${dimNames.head._1}), TensorData.zeros(List(${dimNames.head._1})))\n""")
    sb.append("\n")

    // Forward pass
    var prevExpr = "x"
    for i <- 1 until layers.length do
      val (_, act) = layers(i)
      val isLast = i == layers.length - 1
      val linear = s"(W$i * $prevExpr) + b$i"
      val actName = activationFor(act) match
        case Some(Activation.ReLU)    => "relu"
        case Some(Activation.Sigmoid) => "sigmoid"
        case Some(Activation.Tanh)    => "tanh"
        case Some(Activation.GELU)    => "gelu"
        case Some(Activation.Swish)   => "swish"
        case None                     => ""

      if isLast then
        val expr = if actName.nonEmpty then s"$actName($linear)" else linear
        sb.append(s"val output  = $expr\n")
      else
        val varName = s"hidden$i"
        val expr = if actName.nonEmpty then s"$actName($linear)" else linear
        sb.append(s"val $varName = $expr\n")
        prevExpr = varName

    sb.toString

  /** Return source code for a preset network or custom topology. */
  def sourceFor(name: String): String = name match
    case "twoLayerMlp" =>
      """val inp = Dim("inp", 2)
        |val hid = Dim("hid", 3)
        |val out = Dim("out", 1)
        |
        |val W1 = Ein.Param("W1", List(hid, inp), TensorData.zeros(List(hid, inp)))
        |val b1 = Ein.Param("b1", List(hid), TensorData.zeros(List(hid)))
        |val W2 = Ein.Param("W2", List(out, hid), TensorData.zeros(List(out, hid)))
        |val b2 = Ein.Param("b2", List(out), TensorData.zeros(List(out)))
        |val x  = Ein.Param("x", List(inp), TensorData.zeros(List(inp)))
        |
        |val h = Ein.Activate(Activation.ReLU, (W1 * x) + b1)
        |val output = (W2 * h) + b2""".stripMargin
    case "mnistClassifier" =>
      """val inp = Dim("inp", 784)
        |val h1  = Dim("h1", 128)
        |val h2  = Dim("h2", 64)
        |val out = Dim("out", 10)
        |
        |val W1 = Ein.Param("W1", List(h1, inp), TensorData.zeros(List(h1, inp)))
        |val b1 = Ein.Param("b1", List(h1), TensorData.zeros(List(h1)))
        |val W2 = Ein.Param("W2", List(h2, h1), TensorData.zeros(List(h2, h1)))
        |val b2 = Ein.Param("b2", List(h2), TensorData.zeros(List(h2)))
        |val W3 = Ein.Param("W3", List(out, h2), TensorData.zeros(List(out, h2)))
        |val b3 = Ein.Param("b3", List(out), TensorData.zeros(List(out)))
        |val x  = Ein.Param("x", List(inp), TensorData.zeros(List(inp)))
        |
        |val hidden1 = Ein.Activate(Activation.ReLU, (W1 * x) + b1)
        |val hidden2 = Ein.Activate(Activation.ReLU, (W2 * hidden1) + b2)
        |val output  = (W3 * hidden2) + b3""".stripMargin
    case _ =>
      s"// Unknown preset: $name"
