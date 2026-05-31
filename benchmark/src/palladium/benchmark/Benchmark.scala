package palladium.benchmark

import palladium.ein.*
import palladium.ein.EinDsl.*
import palladium.ein.Block.*

/** Standard MLP benchmark: 3 → 4 → 4 → 1, tanh, MSE loss.
  * Same architecture and data as Karpathy's micrograd example.
  */
object BenchmarkData:
  val xs = Array(
    Array(2.0, 3.0, -1.0),
    Array(3.0, -1.0, 0.5),
    Array(0.5, 1.0, 1.0),
    Array(1.0, 1.0, -1.0)
  )
  val ys = Array(1.0, -1.0, -1.0, 1.0)

  def buildNetwork(seed: Long = 42L): (Ein[Double], Ein[Double], Ein[Double]) =
    given rng: java.util.Random = java.util.Random(seed)

    val inp = 3.dim
    val out = 1.dim
    val x = input[Double](inp)
    val target = input[Double](out)

    val pred =
      x
        >> dense(4, Activation.Tanh, uniform(-1, 1))
        >> dense(4, Activation.Tanh, uniform(-1, 1))
        >> dense(1, Activation.Tanh, uniform(-1, 1))

    val expr = pred.materialize
    val loss = (expr - target).square.sum

    (x, target, loss)

/** Timing utility */
def timed[A](label: String)(f: => A): A =
  val start = System.nanoTime()
  val result = f
  val elapsed = (System.nanoTime() - start) / 1_000_000.0
  println(f"  $label%-30s ${elapsed}%8.1f ms")
  result

/** Benchmark result for a single backend */
case class BenchResult(
    backend: String,
    forwardMs: Double,
    trainingMs: Double,
    finalLoss: Double,
    epochs: Int
)

// --- Backend: Compiled (EinCompile) ---

object CompiledBackend:
  def run(epochs: Int, lr: Double, warmup: Int = 5): BenchResult =
    val (_, _, loss) = BenchmarkData.buildNetwork()
    val mlp = EinCompile.compile(loss)

    // warmup
    for _ <- 0 until warmup do
      mlp.train(BenchmarkData.xs, BenchmarkData.ys, lr = lr, epochs = 10)

    // rebuild fresh for actual benchmark
    val (_, _, loss2) = BenchmarkData.buildNetwork()
    val mlp2 = EinCompile.compile(loss2)

    // benchmark forward
    val forwardStart = System.nanoTime()
    for _ <- 0 until 10_000 do
      for xs <- BenchmarkData.xs do mlp2.forward(xs)
    val forwardMs = (System.nanoTime() - forwardStart) / 1_000_000.0

    // rebuild fresh for training benchmark
    val (_, _, loss3) = BenchmarkData.buildNetwork()
    val mlp3 = EinCompile.compile(loss3)

    val trainStart = System.nanoTime()
    val result = mlp3.train(BenchmarkData.xs, BenchmarkData.ys, lr = lr, epochs = epochs, minLoss = 0.001)
    val trainMs = (System.nanoTime() - trainStart) / 1_000_000.0

    BenchResult("compiled", forwardMs, trainMs, result.finalLoss, result.epochsRun)

// --- Backend: Interpreted (EinEval + EinGrad) ---

object InterpretedBackend:
  def run(epochs: Int, lr: Double): BenchResult =
    import EinDsl.syntax.*

    val (x, target, loss) = BenchmarkData.buildNetwork()

    // benchmark forward (1000 iterations — interpreted is slower)
    val forwardStart = System.nanoTime()
    for _ <- 0 until 1_000 do
      for xs <- BenchmarkData.xs do
        val feed = x.data(xs*)
        EinEval.eval(loss, feed ++ target.data(BenchmarkData.ys(0)))
    val forwardMs = (System.nanoTime() - forwardStart) / 1_000_000.0

    // training via interpreted backward — rebuild params each epoch
    // This is much slower: tree walk per sample per epoch
    val (x2, target2, loss2) = BenchmarkData.buildNetwork()
    var currentLoss = loss2
    var avgLoss = Double.MaxValue

    val trainStart = System.nanoTime()
    var epoch = 0
    while epoch < epochs && avgLoss > 0.001 do
      var totalLoss = 0.0
      val allGrads = scala.collection.mutable.Map[String, Array[Double]]()

      for i <- BenchmarkData.xs.indices do
        val feed = x2.data(BenchmarkData.xs(i)*) ++ target2.data(BenchmarkData.ys(i))
        val lossVal = EinEval.eval(currentLoss, feed)
        totalLoss += lossVal.data(0)
        val grads = EinGrad.backward(currentLoss, feed)
        grads.foreach { (name, td) =>
          val arr = allGrads.getOrElseUpdate(name, new Array[Double](td.data.length))
          for j <- arr.indices do arr(j) += td.data(j)
        }

      // SGD update — rebuild the expression tree with updated params
      val n = BenchmarkData.xs.length
      currentLoss = updateParams(currentLoss, allGrads.toMap, lr, n)
      avgLoss = totalLoss / n
      epoch += 1

    val trainMs = (System.nanoTime() - trainStart) / 1_000_000.0
    BenchResult("interpreted", forwardMs, trainMs, avgLoss, epoch)

  /** Replace Param data in the expression tree with updated values */
  private def updateParams(
      expr: Ein[Double],
      grads: Map[String, Array[Double]],
      lr: Double,
      nSamples: Int
  ): Ein[Double] =
    val scale = lr / nSamples
    expr match
      case Ein.Param(id, dims, data) if grads.contains(id) =>
        val g = grads(id)
        val updated = data.data.clone()
        for i <- updated.indices do updated(i) -= scale * g(i)
        Ein.Param(id, dims, TensorData.fromArray(dims, updated))
      case Ein.Param(_, _, _)           => expr
      case Ein.Input(_, _)              => expr
      case Ein.Fill(_, _)               => expr
      case Ein.Ones(_)                  => expr
      case Ein.Zeros(_)                 => expr
      case Ein.Contract(l, r)           => Ein.Contract(updateParams(l, grads, lr, nSamples), updateParams(r, grads, lr, nSamples))
      case Ein.ElemAdd(l, r)            => Ein.ElemAdd(updateParams(l, grads, lr, nSamples), updateParams(r, grads, lr, nSamples))
      case Ein.ElemSub(l, r)            => Ein.ElemSub(updateParams(l, grads, lr, nSamples), updateParams(r, grads, lr, nSamples))
      case Ein.ElemMul(l, r)            => Ein.ElemMul(updateParams(l, grads, lr, nSamples), updateParams(r, grads, lr, nSamples))
      case Ein.Activate(f, arg)         => Ein.Activate(f, updateParams(arg, grads, lr, nSamples))
      case Ein.ActivateDeriv(f, arg)    => Ein.ActivateDeriv(f, updateParams(arg, grads, lr, nSamples))
      case Ein.ReduceSum(arg, over)     => Ein.ReduceSum(updateParams(arg, grads, lr, nSamples), over)
      case Ein.Broadcast(arg, dims)     => Ein.Broadcast(updateParams(arg, grads, lr, nSamples), dims)
      case Ein.Transpose(arg, perm)     => Ein.Transpose(updateParams(arg, grads, lr, nSamples), perm)
      case Ein.Softmax(arg, dim)        => Ein.Softmax(updateParams(arg, grads, lr, nSamples), dim)
      case Ein.LogSoftmax(arg, dim)     => Ein.LogSoftmax(updateParams(arg, grads, lr, nSamples), dim)
      case Ein.LayerNorm(arg, s, b, d, e) =>
        Ein.LayerNorm(updateParams(arg, grads, lr, nSamples), updateParams(s, grads, lr, nSamples), updateParams(b, grads, lr, nSamples), d, e)
      case Ein.Reshape(arg, dims)       => Ein.Reshape(updateParams(arg, grads, lr, nSamples), dims)
      case Ein.Slice(arg, d, f, t)      => Ein.Slice(updateParams(arg, grads, lr, nSamples), d, f, t)
      case Ein.Gather(table, idx, dim)  => Ein.Gather(updateParams(table, grads, lr, nSamples), idx, dim)
      case Ein.Scatter(src, idx, d, ds) => Ein.Scatter(updateParams(src, grads, lr, nSamples), idx, d, ds)

// --- Main runner ---

@main def runBenchmark(): Unit =
  val epochs = 200
  val lr = 0.1

  println("=" * 60)
  println("Paladium Benchmark: MLP 3 → 4 → 4 → 1, tanh, MSE")
  println("=" * 60)
  println()

  println("--- Compiled backend (EinCompile → flat loops) ---")
  val compiled = timed("total")(CompiledBackend.run(epochs, lr))
  println(f"  forward (10k × 4 samples)   ${compiled.forwardMs}%8.1f ms")
  println(f"  training ($epochs epochs)       ${compiled.trainingMs}%8.1f ms")
  println(f"  final loss                   ${compiled.finalLoss}%.6f")
  println(f"  epochs to converge           ${compiled.epochs}%d")
  println()

  println("--- Interpreted backend (EinEval + EinGrad tree walk) ---")
  val interpreted = timed("total")(InterpretedBackend.run(epochs, lr))
  println(f"  forward (1k × 4 samples)    ${interpreted.forwardMs}%8.1f ms")
  println(f"  training ($epochs epochs)       ${interpreted.trainingMs}%8.1f ms")
  println(f"  final loss                   ${interpreted.finalLoss}%.6f")
  println(f"  epochs to converge           ${interpreted.epochs}%d")
  println()

  println("--- Speedup ---")
  // Normalize forward to same iteration count
  val compiledForwardPer = compiled.forwardMs / 10_000
  val interpretedForwardPer = interpreted.forwardMs / 1_000
  println(f"  forward:  ${interpretedForwardPer / compiledForwardPer}%.1fx faster (compiled)")
  println(f"  training: ${interpreted.trainingMs / compiled.trainingMs}%.1fx faster (compiled)")
  println()
  println("Backends not yet available: blas (forward only), cuda (planned)")
