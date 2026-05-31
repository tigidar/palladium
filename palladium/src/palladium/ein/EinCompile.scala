package palladium.ein

/** Compiles an Ein expression tree representing an MLP with MSE loss
  * into a CompiledMLP that runs forward/backward passes using
  * pre-allocated flat arrays and tight while loops.
  *
  * The Ein algebra is the specification; this interpreter lowers it
  * to something that can run fast.
  */
object EinCompile:

  /** Compiled MLP with pre-allocated mutable buffers.
    * All loops are while-based on flat arrays — no tree walking at runtime.
    */
  final class CompiledMLP(val layerSizes: Array[Int], val activations: Array[Option[Activation]]):
    private val nLayers = layerSizes.length - 1

    // Weights and biases — mutable, updated by SGD
    val w: Array[Array[Double]] =
      Array.tabulate(nLayers)(l => new Array[Double](layerSizes(l + 1) * layerSizes(l)))
    val b: Array[Array[Double]] =
      Array.tabulate(nLayers)(l => new Array[Double](layerSizes(l + 1)))

    // Forward pass: pre-activation (z) and activation (a) per layer
    // a(0) = input, a(l+1) = activate(z(l))
    private val z  = Array.tabulate(nLayers)(l => new Array[Double](layerSizes(l + 1)))
    private val a  = Array.tabulate(nLayers + 1)(l => new Array[Double](layerSizes(l)))

    // Gradient accumulators (summed across samples in a batch)
    private val dw = Array.tabulate(nLayers)(l => new Array[Double](layerSizes(l + 1) * layerSizes(l)))
    private val db = Array.tabulate(nLayers)(l => new Array[Double](layerSizes(l + 1)))

    // Per-layer pre-activation gradient (overwritten each sample, not accumulated)
    private val dz = Array.tabulate(nLayers)(l => new Array[Double](layerSizes(l + 1)))

    /** Forward pass — stores activations for backward. Returns output layer. */
    def forward(input: Array[Double]): Array[Double] =
      System.arraycopy(input, 0, a(0), 0, input.length)
      var l = 0
      while l < nLayers do
        forwardLayer(l)
        l += 1
      a(nLayers)

    private def forwardLayer(l: Int): Unit =
      val inSize  = layerSizes(l)
      val outSize = layerSizes(l + 1)
      val wl = w(l); val bl = b(l); val al = a(l)
      val zl = z(l); val alNext = a(l + 1)
      var i = 0
      while i < outSize do
        var sum    = bl(i)
        val rowOff = i * inSize
        var j = 0
        while j < inSize do
          sum += wl(rowOff + j) * al(j)
          j += 1
        zl(i) = sum
        alNext(i) = applyActivation(activations(l), sum)
        i += 1

    /** Backward pass (MSE loss). Accumulates into dw/db. Returns per-sample loss. */
    def backward(target: Array[Double]): Double =
      // Output gradient from MSE: d(loss)/dz = 2*(pred-target) * (1 - pred^2)
      val outSize = layerSizes(nLayers)
      val aOut    = a(nLayers)
      val dzLast  = dz(nLayers - 1)
      var loss = 0.0
      var i   = 0
      while i < outSize do
        val pred = aOut(i)
        val diff = pred - target(i)
        loss += diff * diff
        dzLast(i) = 2.0 * diff * activationDeriv(activations(nLayers - 1), pred, z(nLayers - 1)(i))
        i += 1

      // Walk layers in reverse
      var l = nLayers - 1
      while l >= 0 do
        backwardLayer(l)
        l -= 1
      loss

    private def backwardLayer(l: Int): Unit =
      val inSize     = layerSizes(l)
      val outSize    = layerSizes(l + 1)
      val wl = w(l); val al = a(l)
      val dzl = dz(l); val dwl = dw(l); val dbl = db(l)

      // Accumulate weight/bias gradients
      var i = 0
      while i < outSize do
        val g      = dzl(i)
        dbl(i)    += g
        val rowOff = i * inSize
        var j = 0
        while j < inSize do
          dwl(rowOff + j) += g * al(j)
          j += 1
        i += 1

      // Propagate gradient through activation to previous layer
      if l > 0 then
        val dzPrev = dz(l - 1)
        val aPrev  = a(l) // post-activation of previous layer
        var j = 0
        while j < inSize do
          var sum = 0.0
          var ii  = 0
          while ii < outSize do
            sum += wl(ii * inSize + j) * dzl(ii)
            ii += 1
          dzPrev(j) = sum * activationDeriv(activations(l - 1), aPrev(j), z(l - 1)(j))
          j += 1

    /** Zero gradient accumulators before a new batch. */
    def zeroGrad(): Unit =
      var l = 0
      while l < nLayers do
        java.util.Arrays.fill(dw(l), 0.0)
        java.util.Arrays.fill(db(l), 0.0)
        l += 1

    /** SGD step: param -= (lr / nSamples) * accumulated_grad */
    def update(lr: Double, nSamples: Int): Unit =
      val scale = lr / nSamples
      var l = 0
      while l < nLayers do
        sgdArray(w(l), dw(l), scale)
        sgdArray(b(l), db(l), scale)
        l += 1

    private def sgdArray(param: Array[Double], grad: Array[Double], scale: Double): Unit =
      var i = 0
      while i < param.length do
        param(i) -= scale * grad(i)
        i += 1

    private def applyActivation(act: Option[Activation], x: Double): Double =
      act match
        case None                      => x
        case Some(Activation.Tanh)     => math.tanh(x)
        case Some(Activation.Sigmoid)  => 1.0 / (1.0 + math.exp(-x))
        case Some(Activation.ReLU)     => math.max(0.0, x)
        case Some(Activation.GELU)     =>
          val inner = math.sqrt(2.0 / math.Pi) * (x + 0.044715 * x * x * x)
          0.5 * x * (1.0 + math.tanh(inner))
        case Some(Activation.Swish)    =>
          x / (1.0 + math.exp(-x))

    private def activationDeriv(act: Option[Activation], activated: Double, preAct: Double): Double =
      act match
        case None                      => 1.0
        case Some(Activation.Tanh)     => 1.0 - activated * activated
        case Some(Activation.Sigmoid)  => activated * (1.0 - activated)
        case Some(Activation.ReLU)     => if preAct > 0.0 then 1.0 else 0.0
        case Some(Activation.GELU)     =>
          val inner = math.sqrt(2.0 / math.Pi) * (preAct + 0.044715 * preAct * preAct * preAct)
          val tanhU = math.tanh(inner)
          val sech2U = 1.0 - tanhU * tanhU
          val duDx = math.sqrt(2.0 / math.Pi) * (1.0 + 3.0 * 0.044715 * preAct * preAct)
          0.5 * (1.0 + tanhU) + 0.5 * preAct * sech2U * duDx
        case Some(Activation.Swish)    =>
          val s = 1.0 / (1.0 + math.exp(-preAct))
          s * (1.0 + preAct * (1.0 - s))
    /** Train this network on the given data using SGD.
      *
      * @param xs      input samples
      * @param ys      target values (one per sample)
      * @param lr      learning rate
      * @param epochs  maximum number of epochs
      * @param minLoss stop early if average loss drops below this
      * @return TrainResult with final loss, epochs run, and convergence status
      */
    def train(
        xs: Array[Array[Double]],
        ys: Array[Double],
        lr: Double = 0.1,
        epochs: Int = 100,
        minLoss: Double = 0.0
    ): TrainResult =
      val n = xs.length
      var avgLoss = Double.MaxValue
      var epoch = 0
      while epoch < epochs && avgLoss > minLoss do
        zeroGrad()
        var totalLoss = 0.0
        var i = 0
        while i < n do
          forward(xs(i))
          totalLoss += backward(Array(ys(i)))
          i += 1
        update(lr, n)
        avgLoss = totalLoss / n
        epoch += 1
      TrainResult(avgLoss, epoch, avgLoss <= minLoss)

  end CompiledMLP

  case class TrainResult(finalLoss: Double, epochsRun: Int, converged: Boolean)

  // --- Compiler: Ein tree → CompiledMLP ---

  /** Walk an Ein expression representing `ReduceSum((pred - target)^2)` and
    * extract the MLP layers into a CompiledMLP with flat-array loops.
    *
    * Expected Ein shape:
    * {{{
    *   ReduceSum(
    *     ElemMul(ElemSub(pred, Input("target")), ElemSub(pred, Input("target"))),
    *     dims)
    * }}}
    * where `pred` is a chain of `Activate(f, ElemAdd(Contract(W, prev), b))` or
    * `ElemAdd(Contract(W, prev), b)` (linear, no activation).
    */
  def compile(expr: Ein[Double]): CompiledMLP =
    val predExpr = recognizeMSE(expr)
    val layers   = extractLayers(predExpr, Nil)
    val sizes    = (layers.head.inputSize +: layers.map(_.outputSize)).toArray
    val acts     = layers.map(_.activation).toArray
    val mlp      = CompiledMLP(sizes, acts)
    var idx = 0
    layers.foreach { layer =>
      System.arraycopy(layer.weightData, 0, mlp.w(idx), 0, layer.weightData.length)
      System.arraycopy(layer.biasData, 0, mlp.b(idx), 0, layer.biasData.length)
      idx += 1
    }
    mlp

  private case class LayerInfo(
      weightData: Array[Double],
      biasData: Array[Double],
      inputSize: Int,
      outputSize: Int,
      activation: Option[Activation]
  )

  /** Peel off MSE loss wrapper, return the prediction expression. */
  private def recognizeMSE(expr: Ein[Double]): Ein[Double] =
    expr match
      case Ein.ReduceSum(Ein.ElemMul(Ein.ElemSub(pred, Ein.Input(_, _)), _), _) =>
        pred
      case _ =>
        throw IllegalArgumentException(
          "Expected MSE loss pattern: ReduceSum(ElemMul(ElemSub(pred, target), ...), dims)"
        )

  /** Tail-recursively walk the layer chain from output back to input,
    * collecting LayerInfo in input-to-output order.
    */
  @annotation.tailrec
  private def extractLayers(expr: Ein[Double], acc: List[LayerInfo]): List[LayerInfo] =
    expr match
      case Ein.Activate(
            f,
            Ein.ElemAdd(
              Ein.Contract(Ein.Param(_, wDims, wData), inner),
              Ein.Param(_, _, bData)
            )
          ) =>
        val outSize = wDims.head.size
        val inSize  = wDims(1).size
        val layer   = LayerInfo(wData.data.clone(), bData.data.clone(), inSize, outSize, Some(f))
        extractLayers(inner, layer :: acc)

      case Ein.ElemAdd(
            Ein.Contract(Ein.Param(_, wDims, wData), inner),
            Ein.Param(_, _, bData)
          ) =>
        val outSize = wDims.head.size
        val inSize  = wDims(1).size
        val layer   = LayerInfo(wData.data.clone(), bData.data.clone(), inSize, outSize, None)
        extractLayers(inner, layer :: acc)

      case Ein.Input(_, _) =>
        acc // base case — layers accumulated in correct order

      case _ =>
        throw IllegalArgumentException(
          s"Expected layer: Activate(f, ElemAdd(Contract(W, prev), b)) or ElemAdd(Contract(W, prev), b)"
        )
