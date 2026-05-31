package palladium.benchmark

import palladium.blas.*
import palladium.ein.*
import palladium.ein.EinDsl.*
import palladium.ein.Block.*

@main def runNativeBenchmark(): Unit =
  val epochs = 200
  val lr = 0.1
  val forwardIters = 10_000

  println("=" * 60)
  println("Paladium Native Benchmark: MLP 3 → 4 → 4 → 1, tanh, MSE")
  println("=" * 60)
  println()

  // --- Build network ---
  given rng: java.util.Random = java.util.Random(42)

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

  val xs = Array(
    Array(2.0, 3.0, -1.0),
    Array(3.0, -1.0, 0.5),
    Array(0.5, 1.0, 1.0),
    Array(1.0, 1.0, -1.0)
  )
  val ys = Array(1.0, -1.0, -1.0, 1.0)

  // --- Compiled backend ---
  println("--- Compiled backend (EinCompile → flat loops) ---")
  locally {
    val mlp = EinCompile.compile(loss)

    val fwdStart = System.nanoTime()
    for _ <- 0 until forwardIters do
      for s <- xs do mlp.forward(s)
    val fwdMs = (System.nanoTime() - fwdStart) / 1_000_000.0

    // rebuild for training
    val (_, _, loss2) = buildNetwork(42L)
    val mlp2 = EinCompile.compile(loss2)
    val trainStart = System.nanoTime()
    val result = mlp2.train(xs, ys, lr = lr, epochs = epochs, minLoss = 0.001)
    val trainMs = (System.nanoTime() - trainStart) / 1_000_000.0

    println(f"  forward ($forwardIters%d × 4 samples)  ${fwdMs}%8.1f ms")
    println(f"  training ($epochs epochs)       ${trainMs}%8.1f ms")
    println(f"  final loss                   ${result.finalLoss}%.6f")
    println(f"  epochs to converge           ${result.epochsRun}%d")
  }
  println()

  // --- BLAS backend (compiled, pre-allocated native buffers) ---
  println("--- BLAS backend (BlasCompile → CBLAS, pre-allocated) ---")
  locally {
    val blasMlp = BlasCompile.compile(loss)

    val fwdStart = System.nanoTime()
    for _ <- 0 until forwardIters do
      for s <- xs do blasMlp.forward(s)
    val fwdMs = (System.nanoTime() - fwdStart) / 1_000_000.0

    // rebuild for training
    val (_, _, loss2) = buildNetwork(42L)
    val blasMlp2 = BlasCompile.compile(loss2)
    val trainStart = System.nanoTime()
    val result = blasMlp2.train(xs, ys, lr = lr, epochs = epochs, minLoss = 0.001)
    val trainMs = (System.nanoTime() - trainStart) / 1_000_000.0

    println(f"  forward ($forwardIters%d × 4 samples)  ${fwdMs}%8.1f ms")
    println(f"  training ($epochs epochs)       ${trainMs}%8.1f ms")
    println(f"  final loss                   ${result.finalLoss}%.6f")
    println(f"  epochs to converge           ${result.epochsRun}%d")
  }
  println()

  // --- CUDA backend ---
  println("--- CUDA backend (cuBLAS on GPU, pre-allocated) ---")
  locally {
    val cudaMlp = palladium.cuda.CudaCompile.compile(loss)

    val fwdStart = System.nanoTime()
    for _ <- 0 until forwardIters do
      for s <- xs do cudaMlp.forward(s)
    val fwdMs = (System.nanoTime() - fwdStart) / 1_000_000.0

    // rebuild for training
    val (_, _, loss2) = buildNetwork(42L)
    val cudaMlp2 = palladium.cuda.CudaCompile.compile(loss2)
    val trainStart = System.nanoTime()
    val result = cudaMlp2.train(xs, ys, lr = lr, epochs = epochs, minLoss = 0.001)
    val trainMs = (System.nanoTime() - trainStart) / 1_000_000.0

    println(f"  forward ($forwardIters%d × 4 samples)  ${fwdMs}%8.1f ms")
    println(f"  training ($epochs epochs)       ${trainMs}%8.1f ms")
    println(f"  final loss                   ${result.finalLoss}%.6f")
    println(f"  epochs to converge           ${result.epochsRun}%d")
  }
  println()

  // --- Interpreted backend ---
  println("--- Interpreted backend (EinEval tree walk) — forward only ---")
  locally {
    val interpIters = 1_000
    val feeds = xs.map(s => x.data(s*) ++ target.data(ys(0)))

    val fwdStart = System.nanoTime()
    for _ <- 0 until interpIters do
      for f <- feeds do EinEval.eval(expr, f)
    val fwdMs = (System.nanoTime() - fwdStart) / 1_000_000.0

    println(f"  forward ($interpIters%d × 4 samples)   ${fwdMs}%8.1f ms")
    println(f"  training                     slow (tree walk per step)")
  }

def buildNetwork(seed: Long): (Ein[Double], Ein[Double], Ein[Double]) =
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
