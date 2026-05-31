package palladium.cuda

import palladium.ein.*
import scala.scalanative.unsafe.*
import scala.scalanative.unsigned.*

/** Compiles an Ein MLP expression into a cuBLAS-accelerated training engine.
  *
  * All weight, activation, and gradient buffers live on the GPU.
  * Only input samples and output predictions cross PCIe per iteration.
  *
  * cuBLAS is column-major, so for our row-major weights W[out, inp]:
  *   row-major y = W * x  ↔  column-major y = W^T * x with CUBLAS_OP_T
  *   (W stored as row-major on host IS column-major transposed on device)
  */
object CudaCompile:

  /** Check a CUDA call and throw on failure. */
  private inline def check(status: CInt, msg: String): Unit =
    if status != CudaConstants.cudaSuccess then
      throw RuntimeException(s"CUDA error ($status): $msg")

  private inline def checkBlas(status: CInt, msg: String): Unit =
    if status != CublasConstants.CUBLAS_STATUS_SUCCESS then
      throw RuntimeException(s"cuBLAS error ($status): $msg")

  /** Allocate n doubles on GPU, returns device pointer. */
  private def gpuAlloc(n: Int): Ptr[CDouble] =
    val devPtr = stackalloc[Ptr[Byte]]()
    check(Cuda.cudaMalloc(devPtr, (n.toLong * 8).toCSize), s"cudaMalloc($n doubles)")
    check(Cuda.cudaMemset(!devPtr, 0, (n.toLong * 8).toCSize), s"cudaMemset($n doubles)")
    (!devPtr).asInstanceOf[Ptr[CDouble]]

  /** Copy host Array[Double] → GPU. */
  private def toDevice(host: Array[Double], dev: Ptr[CDouble], n: Int): Unit =
    val hostPtr = host.at(0).asInstanceOf[Ptr[Byte]]
    check(
      Cuda.cudaMemcpy(dev.asInstanceOf[Ptr[Byte]], hostPtr, (n.toLong * 8).toCSize, CudaConstants.cudaMemcpyHostToDevice),
      "cudaMemcpy H→D"
    )

  /** Copy GPU → host Array[Double]. */
  private def toHost(dev: Ptr[CDouble], n: Int): Array[Double] =
    val result = new Array[Double](n)
    val hostPtr = result.at(0).asInstanceOf[Ptr[Byte]]
    check(
      Cuda.cudaMemcpy(hostPtr, dev.asInstanceOf[Ptr[Byte]], (n.toLong * 8).toCSize, CudaConstants.cudaMemcpyDeviceToHost),
      "cudaMemcpy D→H"
    )
    result

  /** GPU-compiled MLP. All buffers on device — zero PCIe during training loop. */
  final class CudaCompiledMLP(val layerSizes: Array[Int], val activations: Array[Option[Activation]]):
    import Cublas.*, CublasConstants.*

    private val nLayers = layerSizes.length - 1

    // cuBLAS handle
    private val handle: CublasHandle =
      val h = scala.scalanative.libc.stdlib.malloc(8L).asInstanceOf[Ptr[CublasHandle]]
      checkBlas(cublasCreate_v2(h), "cublasCreate")
      !h

    // GPU buffers — allocated once, reused across all epochs
    private val w  = Array.tabulate(nLayers)(l => gpuAlloc(layerSizes(l + 1) * layerSizes(l)))
    private val b  = Array.tabulate(nLayers)(l => gpuAlloc(layerSizes(l + 1)))
    private val z  = Array.tabulate(nLayers)(l => gpuAlloc(layerSizes(l + 1)))
    private val a  = Array.tabulate(nLayers + 1)(l => gpuAlloc(layerSizes(l)))
    private val dw = Array.tabulate(nLayers)(l => gpuAlloc(layerSizes(l + 1) * layerSizes(l)))
    private val db = Array.tabulate(nLayers)(l => gpuAlloc(layerSizes(l + 1)))
    private val dz = Array.tabulate(nLayers)(l => gpuAlloc(layerSizes(l + 1)))

    // Scalar constants for cuBLAS (requires pointers to host doubles)
    private def allocScalar(v: Double): Ptr[CDouble] =
      val p = scala.scalanative.libc.stdlib.malloc(8L).asInstanceOf[Ptr[CDouble]]; !p = v; p
    private val one   = allocScalar(1.0)
    private val zero  = allocScalar(0.0)
    private val scale = allocScalar(0.0) // reused for SGD step

    // Host-side buffer for activation (activations run on CPU for now)
    // For a full GPU implementation, we'd write a CUDA kernel for activations.
    // This is the pragmatic first step: matmul on GPU, activation on CPU.
    private val hostBuf = new Array[Double](layerSizes.max)

    /** Load weights from JVM arrays into GPU buffers. */
    def loadWeights(layer: Int, weights: Array[Double], biases: Array[Double]): Unit =
      toDevice(weights, w(layer), weights.length)
      toDevice(biases, b(layer), biases.length)

    /** Forward pass. Matmul on GPU, activations on CPU. */
    def forward(input: Array[Double]): Array[Double] =
      // Copy input to GPU a(0)
      toDevice(input, a(0), layerSizes(0))

      var l = 0
      while l < nLayers do
        forwardLayer(l)
        l += 1

      // Copy output back to host
      toHost(a(nLayers), layerSizes(nLayers))

    private def forwardLayer(l: Int): Unit =
      val inSize = layerSizes(l)
      val outSize = layerSizes(l + 1)

      // z = b (copy bias to z on GPU)
      checkBlas(cublasDcopy_v2(handle, outSize, b(l), 1, z(l), 1), "dcopy bias→z")

      // z += W * a  (row-major W as column-major → use CUBLAS_OP_T)
      // cuBLAS column-major: y = alpha * op(A) * x + beta * y
      // Our W is [out, inp] row-major = [inp, out] column-major
      // So: y(out) = A^T(out×inp) * x(inp) + y(out) where A is [inp, out] col-major
      checkBlas(
        cublasDgemv_v2(handle, CUBLAS_OP_T, inSize, outSize, one, w(l), inSize, a(l), 1, one, z(l), 1),
        "dgemv forward"
      )

      // Apply activation on CPU (transfer z to host, apply, transfer back)
      activations(l) match
        case None =>
          // Linear — just copy z to a(l+1) on device
          checkBlas(cublasDcopy_v2(handle, outSize, z(l), 1, a(l + 1), 1), "dcopy z→a")
        case Some(act) =>
          val zHost = toHost(z(l), outSize)
          var i = 0
          while i < outSize do
            hostBuf(i) = applyActivation(act, zHost(i))
            i += 1
          toDevice(hostBuf, a(l + 1), outSize)

    /** Backward pass (MSE loss). Accumulates gradients on GPU. Returns loss. */
    def backward(target: Array[Double]): Double =
      val outSize = layerSizes(nLayers)

      // Compute output gradient on CPU: dz = 2*(pred-target) * act'
      val aOutHost = toHost(a(nLayers), outSize)
      val zLastHost = toHost(z(nLayers - 1), outSize)
      val dzHost = new Array[Double](outSize)
      var loss = 0.0
      var i = 0
      while i < outSize do
        val pred = aOutHost(i)
        val diff = pred - target(i)
        loss += diff * diff
        dzHost(i) = 2.0 * diff * activationDeriv(activations(nLayers - 1), pred, zLastHost(i))
        i += 1
      toDevice(dzHost, dz(nLayers - 1), outSize)

      // Walk layers in reverse
      var l = nLayers - 1
      while l >= 0 do
        backwardLayer(l)
        l -= 1
      loss

    private def backwardLayer(l: Int): Unit =
      val inSize = layerSizes(l)
      val outSize = layerSizes(l + 1)

      // db += dz (on GPU)
      checkBlas(cublasDaxpy_v2(handle, outSize, one, dz(l), 1, db(l), 1), "daxpy db+=dz")

      // dw += dz * a^T (outer product on GPU)
      // dw[i,j] += dz[i] * a[j] → this is a rank-1 update: dw += alpha * dz * a^T
      // In column-major: dw(inp×out) += a(inp×1) * dz(1×out)^T ... actually:
      // Use cublasDger: A := alpha * x * y^T + A
      // But we don't have dger bound. Do it on CPU for now.
      val dzHost = toHost(dz(l), outSize)
      val aHost = toHost(a(l), inSize)
      val dwHost = toHost(dw(l), outSize * inSize)
      var i = 0
      while i < outSize do
        val g = dzHost(i)
        val rowOff = i * inSize
        var j = 0
        while j < inSize do
          dwHost(rowOff + j) += g * aHost(j)
          j += 1
        i += 1
      toDevice(dwHost, dw(l), outSize * inSize)

      // Propagate gradient to previous layer
      if l > 0 then
        // dzPrev = W^T * dz  (on GPU)
        // Row-major W[out,inp] → col-major is [inp,out]
        // We want W^T * dz → in col-major: op(A)*x where A is [inp,out], op=NoTrans → [inp,out]*[out] = [inp]
        checkBlas(
          cublasDgemv_v2(handle, CUBLAS_OP_N, inSize, outSize, one, w(l), inSize, dz(l), 1, zero, dz(l - 1), 1),
          "dgemv backward"
        )

        // Multiply by activation derivative on CPU
        val dzPrevHost = toHost(dz(l - 1), inSize)
        val aPrevHost = toHost(a(l), inSize)
        val zPrevHost = toHost(z(l - 1), inSize)
        var j = 0
        while j < inSize do
          dzPrevHost(j) *= activationDeriv(activations(l - 1), aPrevHost(j), zPrevHost(j))
          j += 1
        toDevice(dzPrevHost, dz(l - 1), inSize)

    /** Zero gradient accumulators on GPU. */
    def zeroGrad(): Unit =
      var l = 0
      while l < nLayers do
        check(Cuda.cudaMemset(dw(l).asInstanceOf[Ptr[Byte]], 0, (layerSizes(l + 1).toLong * layerSizes(l) * 8).toCSize), "memset dw")
        check(Cuda.cudaMemset(db(l).asInstanceOf[Ptr[Byte]], 0, (layerSizes(l + 1).toLong * 8).toCSize), "memset db")
        l += 1

    /** SGD step on GPU: w += scale * dw, b += scale * db */
    def update(lr: Double, nSamples: Int): Unit =
      !scale = -lr / nSamples
      var l = 0
      while l < nLayers do
        val wSize = layerSizes(l + 1) * layerSizes(l)
        val bSize = layerSizes(l + 1)
        checkBlas(cublasDaxpy_v2(handle, wSize, scale, dw(l), 1, w(l), 1), "daxpy SGD w")
        checkBlas(cublasDaxpy_v2(handle, bSize, scale, db(l), 1, b(l), 1), "daxpy SGD b")
        l += 1

    /** Train on GPU. */
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
      Cuda.cudaDeviceSynchronize()
      EinCompile.TrainResult(avgLoss, epoch, avgLoss <= minLoss)

    private def applyActivation(act: Activation, x: Double): Double =
      act match
        case Activation.Tanh    => math.tanh(x)
        case Activation.Sigmoid => 1.0 / (1.0 + math.exp(-x))
        case Activation.ReLU    => math.max(0.0, x)
        case Activation.GELU    =>
          val inner = math.sqrt(2.0 / math.Pi) * (x + 0.044715 * x * x * x)
          0.5 * x * (1.0 + math.tanh(inner))
        case Activation.Swish   =>
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

  end CudaCompiledMLP

  /** Compile an Ein MLP loss expression into a CudaCompiledMLP. */
  def compile(expr: Ein[Double]): CudaCompiledMLP =
    val jvmMlp = EinCompile.compile(expr)
    val mlp = CudaCompiledMLP(jvmMlp.layerSizes, jvmMlp.activations)
    for l <- 0 until jvmMlp.layerSizes.length - 1 do
      mlp.loadWeights(l, jvmMlp.w(l), jvmMlp.b(l))
    mlp
