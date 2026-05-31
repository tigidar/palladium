package palladium.ein

class LayerNormSuite extends munit.FunSuite:

  val tolerance = 1e-5
  val epsilon = 1e-5

  // --- helpers ---

  def perturbParam(
      expr: Ein[Double],
      paramId: String,
      elementIndex: Int,
      delta: Double
  ): Ein[Double] =
    expr match
      case Ein.Param(id, dims, data) if id == paramId =>
        val newData = data.data.clone()
        newData(elementIndex) += delta
        Ein.Param(id, dims, TensorData.fromArray(dims, newData))
      case Ein.Param(_, _, _)               => expr
      case Ein.Input(_, _)                  => expr
      case Ein.Fill(_, _)                   => expr
      case Ein.Contract(l, r)               => Ein.Contract(perturbParam(l, paramId, elementIndex, delta), perturbParam(r, paramId, elementIndex, delta))
      case Ein.ElemAdd(l, r)                => Ein.ElemAdd(perturbParam(l, paramId, elementIndex, delta), perturbParam(r, paramId, elementIndex, delta))
      case Ein.ElemSub(l, r)                => Ein.ElemSub(perturbParam(l, paramId, elementIndex, delta), perturbParam(r, paramId, elementIndex, delta))
      case Ein.ElemMul(l, r)                => Ein.ElemMul(perturbParam(l, paramId, elementIndex, delta), perturbParam(r, paramId, elementIndex, delta))
      case Ein.Activate(f, arg)             => Ein.Activate(f, perturbParam(arg, paramId, elementIndex, delta))
      case Ein.ActivateDeriv(f, arg)        => Ein.ActivateDeriv(f, perturbParam(arg, paramId, elementIndex, delta))
      case Ein.ReduceSum(arg, over)         => Ein.ReduceSum(perturbParam(arg, paramId, elementIndex, delta), over)
      case Ein.Broadcast(arg, td)           => Ein.Broadcast(perturbParam(arg, paramId, elementIndex, delta), td)
      case Ein.Transpose(arg, perm)         => Ein.Transpose(perturbParam(arg, paramId, elementIndex, delta), perm)
      case Ein.Softmax(arg, od)             => Ein.Softmax(perturbParam(arg, paramId, elementIndex, delta), od)
      case Ein.LogSoftmax(arg, od)          => Ein.LogSoftmax(perturbParam(arg, paramId, elementIndex, delta), od)
      case Ein.LayerNorm(arg, s, b, od, e)  => Ein.LayerNorm(perturbParam(arg, paramId, elementIndex, delta), perturbParam(s, paramId, elementIndex, delta), perturbParam(b, paramId, elementIndex, delta), od, e)

  def verifyGrad(
      expr: Ein[Double],
      paramId: String,
      paramSize: Int,
      feed: Map[String, TensorData[Double]] = Map.empty
  ): Unit =
    val loss = Ein.ReduceSum(expr, expr.outputDims.map(_.name))
    val grads = EinGrad.backward(loss, feed)
    val grad = grads(paramId)
    for idx <- 0 until paramSize do
      val analytical = grad.data(idx)
      val plusExpr = perturbParam(loss, paramId, idx, epsilon)
      val minusExpr = perturbParam(loss, paramId, idx, -epsilon)
      val numerical = (EinEval.eval(plusExpr, feed).data.sum - EinEval.eval(minusExpr, feed).data.sum) / (2 * epsilon)
      assertEqualsDouble(
        analytical, numerical, tolerance,
        s"Gradient mismatch for $paramId[$idx]: analytical=$analytical, numerical=$numerical"
      )

  // ======================================================
  // Forward tests
  // ======================================================

  test("layernorm forward: output has zero mean over normalized dims") {
    val d = Dim("d", 4)
    val x = Ein.Param("x", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0)))
    val scale = Ein.Param("scale", List(d), TensorData.fromArray(List(d), Array(1.0, 1.0, 1.0, 1.0)))
    val bias = Ein.Param("bias", List(d), TensorData.fromArray(List(d), Array(0.0, 0.0, 0.0, 0.0)))

    val result = EinEval.eval(Ein.LayerNorm(x, scale, bias, List("d"), 1e-5))
    // With scale=1, bias=0, output should have mean ≈ 0
    val mean = result.data.sum / result.data.length
    assertEqualsDouble(mean, 0.0, 1e-6, s"Mean should be ~0, got $mean")
  }

  test("layernorm forward: output has unit variance over normalized dims") {
    val d = Dim("d", 4)
    val x = Ein.Param("x", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0)))
    val scale = Ein.Param("scale", List(d), TensorData.fromArray(List(d), Array(1.0, 1.0, 1.0, 1.0)))
    val bias = Ein.Param("bias", List(d), TensorData.fromArray(List(d), Array(0.0, 0.0, 0.0, 0.0)))

    val result = EinEval.eval(Ein.LayerNorm(x, scale, bias, List("d"), 1e-5))
    val mean = result.data.sum / result.data.length
    val variance = result.data.map(x => (x - mean) * (x - mean)).sum / result.data.length
    assertEqualsDouble(variance, 1.0, 1e-4, s"Variance should be ~1, got $variance")
  }

  test("layernorm forward: scale and bias are applied") {
    val d = Dim("d", 4)
    val x = Ein.Param("x", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0)))
    val scale = Ein.Param("scale", List(d), TensorData.fromArray(List(d), Array(2.0, 2.0, 2.0, 2.0)))
    val bias = Ein.Param("bias", List(d), TensorData.fromArray(List(d), Array(5.0, 5.0, 5.0, 5.0)))

    val result = EinEval.eval(Ein.LayerNorm(x, scale, bias, List("d"), 1e-5))
    // Output should have mean ≈ 5 (bias) and std ≈ 2 (scale)
    val mean = result.data.sum / result.data.length
    assertEqualsDouble(mean, 5.0, 1e-4, s"Mean should be ~5 (bias), got $mean")
  }

  test("layernorm forward: identity with constant input") {
    val d = Dim("d", 3)
    // Constant input → variance = 0, so output = bias (since normalized = 0/eps)
    val x = Ein.Param("x", List(d), TensorData.fromArray(List(d), Array(5.0, 5.0, 5.0)))
    val scale = Ein.Param("scale", List(d), TensorData.fromArray(List(d), Array(1.0, 1.0, 1.0)))
    val bias = Ein.Param("bias", List(d), TensorData.fromArray(List(d), Array(3.0, 3.0, 3.0)))

    val result = EinEval.eval(Ein.LayerNorm(x, scale, bias, List("d"), 1e-5))
    // All inputs are the same → mean = 5, variance ≈ 0 → normalized ≈ 0 → output ≈ bias = 3
    for i <- 0 until 3 do
      assertEqualsDouble(result.data(i), 3.0, 1e-2, s"Element $i should be ~3.0 (bias)")
  }

  test("layernorm forward: preserves output shape") {
    val d = Dim("d", 4)
    val x = Ein.Param("x", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0)))
    val scale = Ein.Param("scale", List(d), TensorData.fromArray(List(d), Array(1.0, 1.0, 1.0, 1.0)))
    val bias = Ein.Param("bias", List(d), TensorData.fromArray(List(d), Array(0.0, 0.0, 0.0, 0.0)))
    val expr = Ein.LayerNorm(x, scale, bias, List("d"), 1e-5)
    assertEquals(expr.outputDims, List(d))
  }

  test("layernorm forward: 2D tensor, normalize over last dim") {
    val batch = Dim("b", 2)
    val features = Dim("d", 3)
    val x = Ein.Param("x", List(batch, features),
      TensorData.fromArray(List(batch, features), Array(
        1.0, 2.0, 3.0,   // sample 0
        10.0, 20.0, 30.0  // sample 1
      )))
    val scale = Ein.Param("scale", List(features),
      TensorData.fromArray(List(features), Array(1.0, 1.0, 1.0)))
    val bias = Ein.Param("bias", List(features),
      TensorData.fromArray(List(features), Array(0.0, 0.0, 0.0)))

    val result = EinEval.eval(Ein.LayerNorm(x, scale, bias, List("d"), 1e-5))
    assertEquals(result.dims, List(batch, features))

    // Each row independently normalized: mean ≈ 0
    val row0sum = result.data(0) + result.data(1) + result.data(2)
    assertEqualsDouble(row0sum, 0.0, 1e-6, s"Row 0 mean should be ~0")
    val row1sum = result.data(3) + result.data(4) + result.data(5)
    assertEqualsDouble(row1sum, 0.0, 1e-6, s"Row 1 mean should be ~0")

    // Both rows produce similar normalized patterns (eps causes slight divergence at different scales)
    assertEqualsDouble(result.data(0), result.data(3), 1e-3, "Normalized patterns should be similar")
    assertEqualsDouble(result.data(1), result.data(4), 1e-3, "Normalized patterns should be similar")
    assertEqualsDouble(result.data(2), result.data(5), 1e-3, "Normalized patterns should be similar")
  }

  test("layernorm forward: known values") {
    val d = Dim("d", 3)
    // Input [1, 2, 3]: mean = 2, var = 2/3
    val x = Ein.Param("x", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val scale = Ein.Param("scale", List(d), TensorData.fromArray(List(d), Array(1.0, 1.0, 1.0)))
    val bias = Ein.Param("bias", List(d), TensorData.fromArray(List(d), Array(0.0, 0.0, 0.0)))
    val eps = 1e-5
    val result = EinEval.eval(Ein.LayerNorm(x, scale, bias, List("d"), eps))

    val mean = 2.0
    val variance = (1.0 + 0.0 + 1.0) / 3.0 // 2/3
    val std = math.sqrt(variance + eps)
    assertEqualsDouble(result.data(0), (1.0 - mean) / std, 1e-6)
    assertEqualsDouble(result.data(1), (2.0 - mean) / std, 1e-6)
    assertEqualsDouble(result.data(2), (3.0 - mean) / std, 1e-6)
  }

  // ======================================================
  // Gradient tests
  // ======================================================

  test("layernorm gradient wrt input matches numerical (1D)") {
    val d = Dim("d", 4)
    val x = Ein.Param("x", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0)))
    val scale = Ein.Param("scale", List(d), TensorData.fromArray(List(d), Array(1.0, 1.0, 1.0, 1.0)))
    val bias = Ein.Param("bias", List(d), TensorData.fromArray(List(d), Array(0.0, 0.0, 0.0, 0.0)))
    val expr = Ein.LayerNorm(x, scale, bias, List("d"), 1e-5)
    verifyGrad(expr, "x", 4)
  }

  test("layernorm gradient wrt scale matches numerical (1D)") {
    val d = Dim("d", 4)
    val x = Ein.Param("x", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0)))
    val scale = Ein.Param("scale", List(d), TensorData.fromArray(List(d), Array(1.0, 1.5, 0.5, 2.0)))
    val bias = Ein.Param("bias", List(d), TensorData.fromArray(List(d), Array(0.0, 0.0, 0.0, 0.0)))
    val expr = Ein.LayerNorm(x, scale, bias, List("d"), 1e-5)
    verifyGrad(expr, "scale", 4)
  }

  test("layernorm gradient wrt bias matches numerical (1D)") {
    val d = Dim("d", 4)
    val x = Ein.Param("x", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0)))
    val scale = Ein.Param("scale", List(d), TensorData.fromArray(List(d), Array(1.0, 1.0, 1.0, 1.0)))
    val bias = Ein.Param("bias", List(d), TensorData.fromArray(List(d), Array(0.1, -0.1, 0.2, 0.0)))
    val expr = Ein.LayerNorm(x, scale, bias, List("d"), 1e-5)
    verifyGrad(expr, "bias", 4)
  }

  test("layernorm gradient wrt all params matches numerical (2D)") {
    val batch = Dim("b", 2)
    val features = Dim("d", 3)
    val x = Ein.Param("x", List(batch, features),
      TensorData.fromArray(List(batch, features), Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    val scale = Ein.Param("scale", List(features),
      TensorData.fromArray(List(features), Array(1.0, 1.5, 0.5)))
    val bias = Ein.Param("bias", List(features),
      TensorData.fromArray(List(features), Array(0.1, -0.1, 0.0)))
    val expr = Ein.LayerNorm(x, scale, bias, List("d"), 1e-5)
    verifyGrad(expr, "x", 6)
    verifyGrad(expr, "scale", 3)
    verifyGrad(expr, "bias", 3)
  }

  test("layernorm gradient in dense >> layernorm pipeline") {
    val features = Dim("j", 2)
    val out = Dim("o", 3)
    val w = Ein.Param("W", List(out, features),
      TensorData.fromArray(List(out, features), Array(0.1, 0.2, -0.3, 0.4, 0.5, -0.1)))
    val b = Ein.Param("b", List(out),
      TensorData.fromArray(List(out), Array(0.1, -0.1, 0.0)))
    val x = Ein.Param("x", List(features),
      TensorData.fromArray(List(features), Array(1.0, -0.5)))
    val scale = Ein.Param("gamma", List(out),
      TensorData.fromArray(List(out), Array(1.0, 1.0, 1.0)))
    val bias = Ein.Param("beta", List(out),
      TensorData.fromArray(List(out), Array(0.0, 0.0, 0.0)))

    val linear = (w * x) + b
    val expr = Ein.LayerNorm(linear, scale, bias, List("o"), 1e-5)
    verifyGrad(expr, "W", 6)
    verifyGrad(expr, "b", 3)
    verifyGrad(expr, "x", 2)
    verifyGrad(expr, "gamma", 3)
    verifyGrad(expr, "beta", 3)
  }

  // ======================================================
  // DSL and Block tests
  // ======================================================

  test("EinDsl layerNorm helper works") {
    val d = Dim("d", 3)
    val x = Ein.Param("x", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val scale = Ein.Param("scale", List(d), TensorData.fromArray(List(d), Array(1.0, 1.0, 1.0)))
    val bias = Ein.Param("bias", List(d), TensorData.fromArray(List(d), Array(0.0, 0.0, 0.0)))
    val r1 = EinEval.eval(EinDsl.layerNorm(x, scale, bias, List("d")))
    val r2 = EinEval.eval(Ein.LayerNorm(x, scale, bias, List("d"), 1e-5))
    for i <- 0 until 3 do
      assertEqualsDouble(r1.data(i), r2.data(i), 1e-10)
  }

  test("Block.layerNorm materializes correctly") {
    val inp = Dim("inp", 4)
    val input = Ein.Input[Double]("x", List(inp))
    val block = Block.layerNorm[Double](List("inp"))
    val net = block.materialize(input)

    // Should be LayerNorm(Input("x"), Param(scale), Param(bias), ...)
    net match
      case Ein.LayerNorm(Ein.Input("x", _), Ein.Param(_, _, _), Ein.Param(_, _, _), List("inp"), _) => ()
      case other => fail(s"Expected LayerNorm(Input, Param, Param, ...), got $other")
  }

  test("Block.layerNorm in pipeline: dense >> layernorm >> softmax") {
    val inp = Dim("inp", 2)
    val input = Ein.Input[Double]("x", List(inp))
    val net = (Block.dense[Double](3, Activation.ReLU) >> Block.layerNorm[Double](List("block0_out")) >> Block.softmax[Double]("block0_out"))
      .materialize(input)

    // Should be Softmax(LayerNorm(Activate(...), scale, bias, ...), ...)
    net match
      case Ein.Softmax(Ein.LayerNorm(Ein.Activate(Activation.ReLU, _), _, _, _, _), _) => ()
      case other => fail(s"Expected Softmax(LayerNorm(ReLU(...),...)), got $other")
  }

  // ======================================================
  // Trace and Mermaid tests
  // ======================================================

  test("EinTrace handles layernorm") {
    val d = Dim("d", 3)
    val x = Ein.Param("x", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val scale = Ein.Param("scale", List(d), TensorData.fromArray(List(d), Array(1.0, 1.0, 1.0)))
    val bias = Ein.Param("bias", List(d), TensorData.fromArray(List(d), Array(0.0, 0.0, 0.0)))
    val expr = Ein.LayerNorm(x, scale, bias, List("d"), 1e-5)
    val graph = EinTrace.forward(expr)

    assertEquals(graph.nodes.size, 4) // x, scale, bias, LayerNorm
    graph.nodes(graph.rootId).kind match
      case EinNodeKind.LayerNorm(List("d"), _) => ()
      case other => fail(s"Expected LayerNorm node kind, got $other")
  }

  test("EinMermaid renders layernorm") {
    val d = Dim("d", 3)
    val x = Ein.Param("x", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val scale = Ein.Param("scale", List(d), TensorData.fromArray(List(d), Array(1.0, 1.0, 1.0)))
    val bias = Ein.Param("bias", List(d), TensorData.fromArray(List(d), Array(0.0, 0.0, 0.0)))
    val expr = Ein.LayerNorm(x, scale, bias, List("d"), 1e-5)
    val graph = EinTrace.forward(expr)
    val mermaid = EinMermaid.renderOps(graph)

    assert(mermaid.contains("layernorm"), s"Mermaid should contain 'layernorm': $mermaid")
  }
