package palladium.ein

class EinCompileSuite extends munit.FunSuite:

  val tolerance = 1e-6

  private def makeMSELoss(
      layerSizes: List[Int],
      activations: List[Option[Activation]],
      seed: Long = 42
  ): (Ein[Double], Array[Array[Double]]) =
    val rng = java.util.Random(seed)
    def init(size: Int): Array[Double] =
      Array.fill(size)((rng.nextDouble() * 2 - 1) * 0.3)

    val dims = layerSizes.zipWithIndex.map { case (size, i) =>
      if i == 0 then Dim("inp", size)
      else Dim(s"h$i", size)
    }

    var prev: Ein[Double] = Ein.Input[Double]("x", List(dims.head))
    val layerPairs = dims.sliding(2).toList.zip(activations)

    for ((pair, act), idx) <- layerPairs.zipWithIndex do
      val inDim = pair(0)
      val outDim = pair(1)
      val wDims = List(outDim, inDim)
      val w = Ein.Param(s"W$idx", wDims, TensorData.fromArray(wDims, init(outDim.size * inDim.size)))
      val b = Ein.Param(s"b$idx", List(outDim), TensorData.fromArray(List(outDim), init(outDim.size)))
      val linear = (w * prev) + b
      prev = act match
        case Some(f) => Ein.Activate(f, linear)
        case None    => linear

    val target = Ein.Input[Double]("target", List(dims.last))
    val diff = prev - target
    val loss = Ein.ReduceSum(diff.elemMul(diff), List(dims.last.name))
    val inputs = Array(init(layerSizes.head), init(layerSizes.last))
    (loss, inputs)

  test("compile recognizes MSE loss pattern") {
    val (loss, _) = makeMSELoss(
      List(3, 4, 1),
      List(Some(Activation.Tanh), None)
    )
    val mlp = EinCompile.compile(loss)
    assertEquals(mlp.layerSizes.toList, List(3, 4, 1))
  }

  test("compile extracts correct activations") {
    val (loss, _) = makeMSELoss(
      List(2, 3, 1),
      List(Some(Activation.ReLU), None)
    )
    val mlp = EinCompile.compile(loss)
    assertEquals(mlp.activations(0), Some(Activation.ReLU))
    assertEquals(mlp.activations(1), None)
  }

  test("compile supports all activation types") {
    for act <- List(Activation.ReLU, Activation.Sigmoid, Activation.Tanh, Activation.GELU, Activation.Swish) do
      val (loss, _) = makeMSELoss(List(2, 1), List(Some(act)))
      val mlp = EinCompile.compile(loss)
      assertEquals(mlp.activations(0), Some(act))
  }

  test("compiled forward produces finite output") {
    val (loss, inputs) = makeMSELoss(
      List(3, 4, 1),
      List(Some(Activation.Tanh), None)
    )
    val mlp = EinCompile.compile(loss)
    val output = mlp.forward(inputs(0))
    assertEquals(output.length, 1)
    assert(output(0).isFinite, s"Output should be finite, got ${output(0)}")
  }

  test("compiled backward produces finite loss") {
    val (loss, inputs) = makeMSELoss(
      List(3, 4, 1),
      List(Some(Activation.Tanh), None)
    )
    val mlp = EinCompile.compile(loss)
    mlp.forward(inputs(0))
    val lossVal = mlp.backward(inputs(1))
    assert(lossVal.isFinite, s"Loss should be finite, got $lossVal")
    assert(lossVal >= 0.0, s"MSE loss should be non-negative, got $lossVal")
  }

  test("zeroGrad resets gradient accumulators") {
    val (loss, inputs) = makeMSELoss(
      List(2, 3, 1),
      List(Some(Activation.Tanh), None)
    )
    val mlp = EinCompile.compile(loss)

    // Accumulate some gradients
    mlp.forward(inputs(0))
    mlp.backward(inputs(1))

    // Zero and verify that a second forward/backward gives the same loss
    val output1 = mlp.forward(inputs(0)).clone()
    mlp.zeroGrad()
    mlp.forward(inputs(0))
    val loss1 = mlp.backward(inputs(1))

    mlp.zeroGrad()
    mlp.forward(inputs(0))
    val loss2 = mlp.backward(inputs(1))

    assertEqualsDouble(loss1, loss2, tolerance, "Same input should give same loss after zeroGrad")
  }

  test("update changes weights") {
    val (loss, inputs) = makeMSELoss(
      List(2, 1),
      List(None)
    )
    val mlp = EinCompile.compile(loss)
    val wBefore = mlp.w(0).clone()

    mlp.zeroGrad()
    mlp.forward(inputs(0))
    mlp.backward(inputs(1))
    mlp.update(0.01, 1)

    val wAfter = mlp.w(0)
    assert(!(wBefore sameElements wAfter), "Weights should change after update")
  }

  test("compile rejects non-MSE expressions") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.zeros[Double](List(d)))
    intercept[IllegalArgumentException] {
      EinCompile.compile(v)
    }
  }

  test("compiled network converges on simple regression") {
    val (loss, _) = makeMSELoss(
      List(2, 4, 1),
      List(Some(Activation.Tanh), None),
      seed = 123
    )
    val mlp = EinCompile.compile(loss)

    // Simple training data: f([1, -1]) = 1, f([-1, 1]) = -1
    val xs = Array(Array(1.0, -1.0), Array(-1.0, 1.0))
    val ys = Array(Array(1.0), Array(-1.0))

    var avgLoss = Double.MaxValue
    for _ <- 0 until 200 do
      mlp.zeroGrad()
      var totalLoss = 0.0
      for i <- xs.indices do
        mlp.forward(xs(i))
        totalLoss += mlp.backward(ys(i))
      mlp.update(0.05, xs.length)
      avgLoss = totalLoss / xs.length

    assert(avgLoss < 0.1, s"Expected convergence, got avgLoss=$avgLoss")
  }
