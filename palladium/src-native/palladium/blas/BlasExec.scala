package palladium.blas

import palladium.codegen.*
import scala.scalanative.unsafe.*
import scala.math

/** Executes a LowProgram using CBLAS for matrix operations and tight
  * while-loops for element-wise operations. All computation uses
  * arena-allocated Ptr[Double] buffers — zero GC pressure, maximum
  * cache efficiency.
  *
  * Entry point: BlasExec.run(prog, paramData, inputData)
  * Returns: output buffers as Array[Double] for interop.
  */
object BlasExec:

  /** Execute a LowProgram and return output arrays.
    *
    * @param prog      the lowered program
    * @param paramData param name → flat Double array
    * @param inputData input name → flat Double array
    * @return one Array[Double] per output in prog.outputs
    */
  def run(
      prog: LowProgram,
      paramData: Map[String, Array[Double]],
      inputData: Map[String, Array[Double]]
  ): Vector[Array[Double]] =
    Zone.acquire { implicit z =>
      // Allocate all buffers
      val buffers = allocateBuffers(prog, paramData, inputData)

      // Execute ops sequentially
      var i = 0
      val ops = prog.ops
      val n = ops.length
      while i < n do
        execOp(ops(i), buffers, prog.indicesData)
        i += 1

      // Extract outputs
      prog.outputs.map { ref =>
        val buf = buffers(ref.id)
        buf.toArray
      }
    }

  // ── Buffer allocation ───────────────────────────────────────────────

  private def allocateBuffers(
      prog: LowProgram,
      paramData: Map[String, Array[Double]],
      inputData: Map[String, Array[Double]]
  )(using Zone): Array[NativeTensor] =
    // Find max buffer ID
    val maxId = {
      val paramIds = prog.params.map(_._1.id)
      val inputIds = prog.inputs.map(_._1.id)
      val tempIds = prog.tempBuffers.map(_.id)
      val outIds = prog.outputs.map(_.id)
      (paramIds ++ inputIds ++ tempIds ++ outIds).max
    }
    val buffers = new Array[NativeTensor](maxId + 1)

    // Allocate param buffers and load data
    for (ref, name) <- prog.params do
      val data = paramData.getOrElse(name, new Array[Double](ref.totalSize))
      buffers(ref.id) = NativeTensor.fromArray(ref.shape, data)

    // Allocate input buffers and load data
    for (ref, name) <- prog.inputs do
      val data = inputData.getOrElse(name, new Array[Double](ref.totalSize))
      buffers(ref.id) = NativeTensor.fromArray(ref.shape, data)

    // Allocate temp/output buffers (zero-initialized)
    for ref <- prog.tempBuffers do
      if buffers(ref.id) == null then
        buffers(ref.id) = NativeTensor.zeros(ref.shape)

    buffers

  // ── Operation dispatch ──────────────────────────────────────────────

  private def execOp(op: LowOp, buffers: Array[NativeTensor], indicesData: Map[Int, Array[Int]]): Unit =
    op match
      case LowOp.MatMul(out, left, right, _, leftFreeCount, rightFreeCount, contractSize) =>
        execMatMul(buffers(out.id), buffers(left.id), buffers(right.id),
                   leftFreeCount, rightFreeCount, contractSize)

      case LowOp.ElemBinary(out, left, right, bop) =>
        execElemBinary(buffers(out.id), buffers(left.id), buffers(right.id), bop)

      case LowOp.Activate(out, input, f) =>
        execActivate(buffers(out.id), buffers(input.id), f)

      case LowOp.ActivateDeriv(out, input, f) =>
        execActivateDeriv(buffers(out.id), buffers(input.id), f)

      case LowOp.ReduceSum(out, input, keepDims, reduceDims) =>
        execReduceSum(buffers(out.id), buffers(input.id), reduceDims)

      case LowOp.Broadcast(out, input, _) =>
        execBroadcast(buffers(out.id), buffers(input.id))

      case LowOp.Copy(out, input) =>
        execCopy(buffers(out.id), buffers(input.id))

      case LowOp.FillOnes(out) =>
        buffers(out.id).fill(1.0)

      case LowOp.FillZeros(out) =>
        buffers(out.id).zero()

      case LowOp.Softmax(out, input, overDimSize) =>
        execSoftmax(buffers(out.id), buffers(input.id), overDimSize)

      case LowOp.LogSoftmax(out, input, overDimSize) =>
        execLogSoftmax(buffers(out.id), buffers(input.id), overDimSize)

      case LowOp.LayerNorm(out, input, scale, bias, normSize, eps) =>
        execLayerNorm(buffers(out.id), buffers(input.id),
                      buffers(scale.id), buffers(bias.id), normSize, eps)

      case LowOp.Reshape(out, input, _) =>
        execCopy(buffers(out.id), buffers(input.id))

      case LowOp.Gather(out, table, indicesId, indicesSize) =>
        execGather(buffers(out.id), buffers(table.id), indicesData(indicesId), indicesSize)

      case LowOp.Scatter(out, src, indicesId, indicesSize) =>
        execScatter(buffers(out.id), buffers(src.id), indicesData(indicesId), indicesSize)

  // ── MatMul via CBLAS ────────────────────────────────────────────────

  /** Dispatches to dgemm (matrix*matrix), dgemv (matrix*vector), or
    * ddot (vector*vector) based on the free dimension counts.
    *
    * Layout: row-major. For W(M×K) * x(K×N):
    *   M = leftFreeSize, N = rightFreeSize, K = contractSize
    */
  private def execMatMul(
      out: NativeTensor,
      left: NativeTensor,
      right: NativeTensor,
      leftFreeCount: Int,
      rightFreeCount: Int,
      contractSize: Int
  ): Unit =
    val leftFreeSize = if leftFreeCount == 0 then 1 else left.totalSize / contractSize
    val rightFreeSize = if rightFreeCount == 0 then 1 else right.totalSize / contractSize

    if leftFreeCount > 0 && rightFreeCount > 0 then
      // Matrix-matrix: C(M×N) = A(M×K) * B(K×N)
      Cblas.cblas_dgemm(
        CblasConstants.CblasRowMajor,
        CblasConstants.CblasNoTrans,
        CblasConstants.CblasNoTrans,
        leftFreeSize,   // M
        rightFreeSize,   // N
        contractSize,    // K
        1.0,             // alpha
        left.data,
        contractSize,    // lda = K
        right.data,
        rightFreeSize,   // ldb = N
        0.0,             // beta
        out.data,
        rightFreeSize    // ldc = N
      )
    else if leftFreeCount > 0 && rightFreeCount == 0 then
      // Matrix-vector: y(M) = A(M×K) * x(K)
      Cblas.cblas_dgemv(
        CblasConstants.CblasRowMajor,
        CblasConstants.CblasNoTrans,
        leftFreeSize,    // M
        contractSize,    // N (cols of A = size of x)
        1.0,             // alpha
        left.data,
        contractSize,    // lda
        right.data,
        1,               // incX
        0.0,             // beta
        out.data,
        1                // incY
      )
    else if leftFreeCount == 0 && rightFreeCount > 0 then
      // Vector-matrix: y(N) = x^T(K) * B(K×N) = B^T(N×K) * x(K)
      // Use dgemv with transpose
      Cblas.cblas_dgemv(
        CblasConstants.CblasRowMajor,
        CblasConstants.CblasTrans,
        contractSize,    // M (rows of B)
        rightFreeSize,   // N (cols of B)
        1.0,
        right.data,
        rightFreeSize,   // lda
        left.data,
        1,
        0.0,
        out.data,
        1
      )
    else
      // Dot product: scalar = x . y
      val result = Cblas.cblas_ddot(contractSize, left.data, 1, right.data, 1)
      out(0) = result

  // ── Element-wise binary via tight loop ──────────────────────────────

  private def execElemBinary(
      out: NativeTensor,
      left: NativeTensor,
      right: NativeTensor,
      op: BinaryOp
  ): Unit =
    val outSize = out.totalSize
    val leftSize = left.totalSize
    val rightSize = right.totalSize
    val od = out.data
    val ld = left.data
    val rd = right.data

    // Fast path: same-size, no broadcast needed
    if leftSize == outSize && rightSize == outSize then
      op match
        case BinaryOp.Add =>
          // out = left; out += right  (BLAS: dcopy + daxpy)
          Cblas.cblas_dcopy(outSize, ld, 1, od, 1)
          Cblas.cblas_daxpy(outSize, 1.0, rd, 1, od, 1)
        case BinaryOp.Sub =>
          // out = left; out -= right
          Cblas.cblas_dcopy(outSize, ld, 1, od, 1)
          Cblas.cblas_daxpy(outSize, -1.0, rd, 1, od, 1)
        case BinaryOp.Mul =>
          // No BLAS for elem-wise mul; tight loop
          var i = 0
          while i < outSize do
            od(i) = ld(i) * rd(i)
            i += 1
    else
      // Broadcast path: use modulo indexing
      var i = 0
      op match
        case BinaryOp.Add =>
          while i < outSize do
            od(i) = ld(i % leftSize) + rd(i % rightSize)
            i += 1
        case BinaryOp.Sub =>
          while i < outSize do
            od(i) = ld(i % leftSize) - rd(i % rightSize)
            i += 1
        case BinaryOp.Mul =>
          while i < outSize do
            od(i) = ld(i % leftSize) * rd(i % rightSize)
            i += 1

  // ── Activations via tight loops ─────────────────────────────────────

  private def execActivate(out: NativeTensor, input: NativeTensor, f: ActivationType): Unit =
    val n = out.totalSize
    val od = out.data
    val id = input.data
    var i = 0
    f match
      case ActivationType.ReLU =>
        while i < n do
          val v = id(i)
          od(i) = if v > 0.0 then v else 0.0
          i += 1
      case ActivationType.Sigmoid =>
        while i < n do
          od(i) = 1.0 / (1.0 + math.exp(-id(i)))
          i += 1
      case ActivationType.Tanh =>
        while i < n do
          od(i) = math.tanh(id(i))
          i += 1
      case ActivationType.GELU =>
        val sqrtCoeff = math.sqrt(2.0 / math.Pi)
        while i < n do
          val v = id(i)
          od(i) = 0.5 * v * (1.0 + math.tanh(sqrtCoeff * (v + 0.044715 * v * v * v)))
          i += 1
      case ActivationType.Swish =>
        while i < n do
          val v = id(i)
          od(i) = v / (1.0 + math.exp(-v))
          i += 1

  private def execActivateDeriv(out: NativeTensor, input: NativeTensor, f: ActivationType): Unit =
    val n = out.totalSize
    val od = out.data
    val id = input.data
    var i = 0
    f match
      case ActivationType.ReLU =>
        while i < n do
          od(i) = if id(i) > 0.0 then 1.0 else 0.0
          i += 1
      case ActivationType.Sigmoid =>
        while i < n do
          val s = 1.0 / (1.0 + math.exp(-id(i)))
          od(i) = s * (1.0 - s)
          i += 1
      case ActivationType.Tanh =>
        while i < n do
          val t = math.tanh(id(i))
          od(i) = 1.0 - t * t
          i += 1
      case ActivationType.GELU =>
        val sqrtCoeff = math.sqrt(2.0 / math.Pi)
        while i < n do
          val v = id(i)
          val u = sqrtCoeff * (v + 0.044715 * v * v * v)
          val t = math.tanh(u)
          val du = sqrtCoeff * (1.0 + 3.0 * 0.044715 * v * v)
          od(i) = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * du
          i += 1
      case ActivationType.Swish =>
        while i < n do
          val v = id(i)
          val s = 1.0 / (1.0 + math.exp(-v))
          od(i) = s * (1.0 + v * (1.0 - s))
          i += 1

  // ── ReduceSum via tight loop ────────────────────────────────────────

  private def execReduceSum(
      out: NativeTensor,
      input: NativeTensor,
      reduceDims: List[Int]
  ): Unit =
    val outSize = out.totalSize
    val reduceSize = if reduceDims.isEmpty then 1 else reduceDims.product
    val od = out.data
    val id = input.data

    out.zero()
    var i = 0
    while i < outSize do
      var sum = 0.0
      var j = 0
      while j < reduceSize do
        sum += id(i * reduceSize + j)
        j += 1
      od(i) = sum
      i += 1

  // ── Broadcast via tight loop ────────────────────────────────────────

  private def execBroadcast(out: NativeTensor, input: NativeTensor): Unit =
    val outSize = out.totalSize
    val inSize = input.totalSize
    val od = out.data
    val id = input.data
    var i = 0
    while i < outSize do
      od(i) = id(i % inSize)
      i += 1

  // ── Copy via BLAS ───────────────────────────────────────────────────

  private def execCopy(out: NativeTensor, input: NativeTensor): Unit =
    val n = out.totalSize.min(input.totalSize)
    Cblas.cblas_dcopy(n, input.data, 1, out.data, 1)

  // ── Softmax (numerically stable) ───────────────────────────────────

  private def execSoftmax(out: NativeTensor, input: NativeTensor, overDimSize: Int): Unit =
    val totalSize = out.totalSize
    val batchSize = totalSize / overDimSize
    val od = out.data
    val id = input.data

    var b = 0
    while b < batchSize do
      val base = b * overDimSize
      // Find max for numerical stability
      var maxv = id(base)
      var i = 1
      while i < overDimSize do
        val v = id(base + i)
        if v > maxv then maxv = v
        i += 1
      // Exp and sum
      var sum = 0.0
      i = 0
      while i < overDimSize do
        val e = math.exp(id(base + i) - maxv)
        od(base + i) = e
        sum += e
        i += 1
      // Normalize
      i = 0
      while i < overDimSize do
        od(base + i) = od(base + i) / sum
        i += 1
      b += 1

  // ── LogSoftmax (numerically stable) ────────────────────────────────

  private def execLogSoftmax(out: NativeTensor, input: NativeTensor, overDimSize: Int): Unit =
    val totalSize = out.totalSize
    val batchSize = totalSize / overDimSize
    val od = out.data
    val id = input.data

    var b = 0
    while b < batchSize do
      val base = b * overDimSize
      var maxv = id(base)
      var i = 1
      while i < overDimSize do
        val v = id(base + i)
        if v > maxv then maxv = v
        i += 1
      var logsum = 0.0
      i = 0
      while i < overDimSize do
        logsum += math.exp(id(base + i) - maxv)
        i += 1
      logsum = math.log(logsum)
      i = 0
      while i < overDimSize do
        od(base + i) = (id(base + i) - maxv) - logsum
        i += 1
      b += 1

  // ── LayerNorm ──────────────────────────────────────────────────────

  private def execLayerNorm(
      out: NativeTensor,
      input: NativeTensor,
      scale: NativeTensor,
      bias: NativeTensor,
      normSize: Int,
      eps: Double
  ): Unit =
    val totalSize = out.totalSize
    val batchSize = totalSize / normSize
    val od = out.data
    val id = input.data
    val sd = scale.data
    val bd = bias.data

    var b = 0
    while b < batchSize do
      val base = b * normSize
      // Mean
      var mean = 0.0
      var i = 0
      while i < normSize do
        mean += id(base + i)
        i += 1
      mean /= normSize
      // Variance
      var variance = 0.0
      i = 0
      while i < normSize do
        val d = id(base + i) - mean
        variance += d * d
        i += 1
      variance /= normSize
      val invStd = 1.0 / math.sqrt(variance + eps)
      // Normalize, scale, shift
      i = 0
      while i < normSize do
        od(base + i) = (id(base + i) - mean) * invStd * sd(i) + bd(i)
        i += 1
      b += 1

  // ── Gather (embedding lookup) ──────────────────────────────────────

  private def execGather(
      out: NativeTensor,
      table: NativeTensor,
      indices: Array[Int],
      indicesSize: Int
  ): Unit =
    val od = out.data
    val td = table.data
    val embDim = table.totalSize / (if table.shape.nonEmpty then table.shape.head else 1)

    var i = 0
    while i < indicesSize do
      val idx = indices(i)
      var j = 0
      while j < embDim do
        od(i * embDim + j) = td(idx * embDim + j)
        j += 1
      i += 1

  // ── Scatter (embedding gradient accumulation) ──────────────────────

  private def execScatter(
      out: NativeTensor,
      src: NativeTensor,
      indices: Array[Int],
      indicesSize: Int
  ): Unit =
    val od = out.data
    val sd = src.data
    val embDim = if src.shape.size > 1 then src.shape.last else 1

    out.zero()
    var i = 0
    while i < indicesSize do
      val idx = indices(i)
      var j = 0
      while j < embDim do
        od(idx * embDim + j) = od(idx * embDim + j) + sd(i * embDim + j)
        j += 1
      i += 1
