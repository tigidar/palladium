package palladium.ein

class EinNeuralNetSuite extends munit.FunSuite:

  test("compile Ein MLP to optimized loops and train via SGD") {
    // Training data (from test-net.md)
    val xs = Array(
      Array(2.0, 3.0, -1.0),
      Array(3.0, -1.0, 0.5),
      Array(0.5, 1.0, 1.0),
      Array(1.0, 1.0, -1.0)
    )
    val ys = Array(1.0, -1.0, -1.0, 1.0)
    val n = xs.length

    // --- 1. Specify the network using the Ein algebra ---
    // Architecture: 3 -> 4 -> 4 -> 1, all tanh, MSE loss
    val inp = Dim("j", 3)
    val h1  = Dim("h1", 4)
    val h2  = Dim("h2", 4)
    val out = Dim("o", 1)

    val rng = java.util.Random(42)
    def init(rows: Int, cols: Int): Array[Double] =
      Array.fill(rows * cols)((rng.nextDouble() * 2 - 1) * 0.5)

    val W0 = Ein.Param("W0", List(h1, inp), TensorData.fromArray(List(h1, inp), init(4, 3)))
    val B0 = Ein.Param("b0", List(h1), TensorData.fromArray(List(h1), Array.fill(4)(0.0)))
    val W1 = Ein.Param("W1", List(h2, h1), TensorData.fromArray(List(h2, h1), init(4, 4)))
    val B1 = Ein.Param("b1", List(h2), TensorData.fromArray(List(h2), Array.fill(4)(0.0)))
    val W2 = Ein.Param("W2", List(out, h2), TensorData.fromArray(List(out, h2), init(1, 4)))
    val B2 = Ein.Param("b2", List(out), TensorData.fromArray(List(out), Array.fill(1)(0.0)))

    val x      = Ein.Input[Double]("x", List(inp))
    val target = Ein.Input[Double]("target", List(out))

    val layer0 = EinDsl.tanh((W0 * x) + B0)
    val layer1 = EinDsl.tanh((W1 * layer0) + B1)
    val pred   = EinDsl.tanh((W2 * layer1) + B2)
    val diff   = pred - target
    val loss   = Ein.ReduceSum(diff.elemMul(diff), List("o"))

    // --- 2. Compile the algebra down to flat-array loops ---
    val mlp = EinCompile.compile(loss)

    // --- 3. Train using the compiled network ---
    val lr        = 0.1
    val maxEpochs = 500
    var avgLoss   = Double.MaxValue
    var epoch     = 0

    while epoch < maxEpochs && avgLoss >= 0.005 do
      mlp.zeroGrad()
      var totalLoss = 0.0
      var i = 0
      while i < n do
        mlp.forward(xs(i))
        totalLoss += mlp.backward(Array(ys(i)))
        i += 1
      mlp.update(lr, n)
      avgLoss = totalLoss / n
      epoch += 1

    assert(avgLoss < 0.01, s"Expected loss < 0.01 after $epoch epochs, got $avgLoss")
    println(s"  Converged in $epoch epochs, avgLoss=$avgLoss")

    // --- 4. Verify predictions match targets ---
    for i <- 0 until n do
      val prediction = mlp.forward(xs(i))
      assertEqualsDouble(prediction(0), ys(i), 0.15,
        s"Sample $i: expected ${ys(i)}, got ${prediction(0)}")
  }
