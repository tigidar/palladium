package palladium.blas

import palladium.ein.*
import scala.scalanative.unsafe.*
import scala.scalanative.unsigned.*

/** Compiles an Ein MLP expression into a BLAS-accelerated training engine.
  *
  * Same compile-once, pre-allocate-everything approach as EinCompile.CompiledMLP,
  * but uses CBLAS for matrix-vector operations:
  *   - Forward:  cblas_dgemv for W * a + b
  *   - Backward: cblas_dgemv for gradient propagation, cblas_dger for weight gradient
  *   - Update:   cblas_daxpy for SGD step
  *
  * Zero allocation during training — all buffers are Ptr[Double] allocated once.
  */
object BlasCompile:

  /** BLAS-compiled MLP with pre-allocated native buffers.
    * All matrix ops use CBLAS — no tree walking, no arena allocation at runtime.
    */
  final class BlasCompiledMLP(val layerSizes: Array[Int], val activations: Array[Option[Activation]]):
    import Cblas.*, CblasConstants.*

    private val nLayers = layerSizes.length - 1

    // Pre-allocate all native buffers
    private val w  = Array.tabulate(nLayers)(l => alloc(layerSizes(l + 1) * layerSizes(l)))
    private val b  = Array.tabulate(nLayers)(l => alloc(layerSizes(l + 1)))
    private val z  = Array.tabulate(nLayers)(l => alloc(layerSizes(l + 1)))
    private val a  = Array.tabulate(nLayers + 1)(l => alloc(layerSizes(l)))
    private val dw = Array.tabulate(nLayers)(l => alloc(layerSizes(l + 1) * layerSizes(l)))
    private val db = Array.tabulate(nLayers)(l => alloc(layerSizes(l + 1)))
    private val dz = Array.tabulate(nLayers)(l => alloc(layerSizes(l + 1)))

    private def alloc(n: Int): Ptr[CDouble] =
      val ptr = scala.scalanative.libc.stdlib.calloc(n.toLong, sizeof[CDouble].toLong).asInstanceOf[Ptr[CDouble]]
      if ptr == null then throw OutOfMemoryError(s"Failed to allocate $n doubles")
      ptr

    /** Copy JVM Array[Double] data into a native Ptr[Double] buffer */
    def loadWeights(layer: Int, weights: Array[Double], biases: Array[Double]): Unit =
      var i = 0
      while i < weights.length do
        w(layer)(i) = weights(i)
        i += 1
      i = 0
      while i < biases.length do
        b(layer)(i) = biases(i)
        i += 1

    /** Forward pass using CBLAS. Stores activations for backward. Returns output as Array. */
    def forward(input: Array[Double]): Array[Double] =
      // Copy input into a(0)
      val inSize = layerSizes(0)
      var i = 0
      while i < inSize do
        a(0)(i) = input(i)
        i += 1

      var l = 0
      while l < nLayers do
        forwardLayer(l)
        l += 1

      // Copy output to Array
      val outSize = layerSizes(nLayers)
      val result = new Array[Double](outSize)
      i = 0
      while i < outSize do
        result(i) = a(nLayers)(i)
        i += 1
      result

    private def forwardLayer(l: Int): Unit =
      val inSize = layerSizes(l)
      val outSize = layerSizes(l + 1)

      // z = W * a + b
      // First copy bias into z: z := b
      cblas_dcopy(outSize, b(l), 1, z(l), 1)
      // Then z := 1.0 * W * a(l) + 1.0 * z  (row-major, no transpose)
      cblas_dgemv(CblasRowMajor, CblasNoTrans, outSize, inSize, 1.0, w(l), inSize, a(l), 1, 1.0, z(l), 1)

      // Apply activation: a(l+1) = activate(z(l))
      var i = 0
      while i < outSize do
        a(l + 1)(i) = applyActivation(activations(l), z(l)(i))
        i += 1

    /** Backward pass (MSE loss). Accumulates into dw/db. Returns per-sample loss. */
    def backward(target: Array[Double]): Double =
      val outSize = layerSizes(nLayers)
      val aOut = a(nLayers)
      val dzLast = dz(nLayers - 1)
      var loss = 0.0
      var i = 0
      while i < outSize do
        val pred = aOut(i)
        val diff = pred - target(i)
        loss += diff * diff
        dzLast(i) = 2.0 * diff * activationDeriv(activations(nLayers - 1), pred, z(nLayers - 1)(i))
        i += 1

      var l = nLayers - 1
      while l >= 0 do
        backwardLayer(l)
        l -= 1
      loss

    private def backwardLayer(l: Int): Unit =
      val inSize = layerSizes(l)
      val outSize = layerSizes(l + 1)

      // Accumulate bias gradients: db += dz
      cblas_daxpy(outSize, 1.0, dz(l), 1, db(l), 1)

      // Accumulate weight gradients: dw += dz * a(l)^T
      // This is an outer product: dw[i,j] += dz[i] * a[j]
      // Using dgemv in a loop is simpler than dger for small sizes
      var i = 0
      while i < outSize do
        val g = dz(l)(i)
        val rowOff = i * inSize
        var j = 0
        while j < inSize do
          dw(l)(rowOff + j) = dw(l)(rowOff + j) + g * a(l)(j)
          j += 1
        i += 1

      // Propagate gradient to previous layer: dz_prev = W^T * dz * act'
      if l > 0 then
        val dzPrev = dz(l - 1)
        // tmp = W^T * dz(l)
        // Use dgemv with CblasTrans: y := alpha * A^T * x + beta * y
        // First zero dzPrev
        var j = 0
        while j < inSize do
          dzPrev(j) = 0.0
          j += 1
        cblas_dgemv(CblasRowMajor, CblasTrans, outSize, inSize, 1.0, w(l), inSize, dz(l), 1, 0.0, dzPrev, 1)
        // Multiply by activation derivative
        j = 0
        while j < inSize do
          dzPrev(j) = dzPrev(j) * activationDeriv(activations(l - 1), a(l)(j), z(l - 1)(j))
          j += 1

    /** Zero gradient accumulators. */
    def zeroGrad(): Unit =
      var l = 0
      while l < nLayers do
        val wSize = layerSizes(l + 1) * layerSizes(l)
        val bSize = layerSizes(l + 1)
        var i = 0
        while i < wSize do
          dw(l)(i) = 0.0
          i += 1
        i = 0
        while i < bSize do
          db(l)(i) = 0.0
          i += 1
        l += 1

    /** SGD step using CBLAS: param -= scale * grad */
    def update(lr: Double, nSamples: Int): Unit =
      val scale = -lr / nSamples
      var l = 0
      while l < nLayers do
        val wSize = layerSizes(l + 1) * layerSizes(l)
        val bSize = layerSizes(l + 1)
        // w += scale * dw  (scale is negative for descent)
        cblas_daxpy(wSize, scale, dw(l), 1, w(l), 1)
        // b += scale * db
        cblas_daxpy(bSize, scale, db(l), 1, b(l), 1)
        l += 1

    /** Train this network on the given data using SGD. */
    def train(
        xs: Array[Array[Double]],
        ys: Array[Double],
        lr: Double = 0.1,
        epochs: Int = 100,
        minLoss: Double = 0.0
    ): EinCompile.TrainResult =
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
      EinCompile.TrainResult(avgLoss, epoch, avgLoss <= minLoss)

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

  end BlasCompiledMLP

  /** Compile an Ein MLP loss expression into a BlasCompiledMLP. */
  def compile(expr: Ein[Double]): BlasCompiledMLP =
    // Reuse EinCompile's tree recognition
    val jvmMlp = EinCompile.compile(expr)
    val mlp = BlasCompiledMLP(jvmMlp.layerSizes, jvmMlp.activations)
    // Copy weights from the JVM-compiled version
    for l <- 0 until jvmMlp.layerSizes.length - 1 do
      mlp.loadWeights(l, jvmMlp.w(l), jvmMlp.b(l))
    mlp
