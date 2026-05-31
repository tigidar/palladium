package palladium.ein

class EinSoftmaxSuite extends munit.FunSuite:

  val tolerance = 1e-5
  val epsilon = 1e-5

  // --- helpers (same pattern as EinGradSuite) ---

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
      case Ein.Param(_, _, _)        => expr
      case Ein.Input(_, _)           => expr
      case Ein.Fill(_, _)            => expr
      case Ein.Contract(l, r)        => Ein.Contract(perturbParam(l, paramId, elementIndex, delta), perturbParam(r, paramId, elementIndex, delta))
      case Ein.ElemAdd(l, r)         => Ein.ElemAdd(perturbParam(l, paramId, elementIndex, delta), perturbParam(r, paramId, elementIndex, delta))
      case Ein.ElemSub(l, r)         => Ein.ElemSub(perturbParam(l, paramId, elementIndex, delta), perturbParam(r, paramId, elementIndex, delta))
      case Ein.ElemMul(l, r)         => Ein.ElemMul(perturbParam(l, paramId, elementIndex, delta), perturbParam(r, paramId, elementIndex, delta))
      case Ein.Activate(f, arg)      => Ein.Activate(f, perturbParam(arg, paramId, elementIndex, delta))
      case Ein.ActivateDeriv(f, arg) => Ein.ActivateDeriv(f, perturbParam(arg, paramId, elementIndex, delta))
      case Ein.ReduceSum(arg, over)  => Ein.ReduceSum(perturbParam(arg, paramId, elementIndex, delta), over)
      case Ein.Broadcast(arg, td)    => Ein.Broadcast(perturbParam(arg, paramId, elementIndex, delta), td)
      case Ein.Transpose(arg, perm)  => Ein.Transpose(perturbParam(arg, paramId, elementIndex, delta), perm)
      case Ein.Softmax(arg, od)      => Ein.Softmax(perturbParam(arg, paramId, elementIndex, delta), od)
      case Ein.LogSoftmax(arg, od)   => Ein.LogSoftmax(perturbParam(arg, paramId, elementIndex, delta), od)
      case Ein.LayerNorm(arg, s, b, od, e) => Ein.LayerNorm(perturbParam(arg, paramId, elementIndex, delta), perturbParam(s, paramId, elementIndex, delta), perturbParam(b, paramId, elementIndex, delta), od, e)

  def numericalGrad(
      expr: Ein[Double],
      paramId: String,
      elementIndex: Int,
      feed: Map[String, TensorData[Double]] = Map.empty
  ): Double =
    val plusExpr = perturbParam(expr, paramId, elementIndex, epsilon)
    val plusVal = EinEval.eval(plusExpr, feed)
    val plusLoss = plusVal.data.sum

    val minusExpr = perturbParam(expr, paramId, elementIndex, -epsilon)
    val minusVal = EinEval.eval(minusExpr, feed)
    val minusLoss = minusVal.data.sum

    (plusLoss - minusLoss) / (2 * epsilon)

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
      val numerical = numericalGrad(loss, paramId, idx, feed)
      assertEqualsDouble(
        analytical, numerical, tolerance,
        s"Gradient mismatch for $paramId[$idx]: analytical=$analytical, numerical=$numerical"
      )

  // --- Softmax forward tests ---

  test("softmax forward: values sum to 1") {
    val d = Dim("d", 4)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0)))
    val result = EinEval.eval(Ein.Softmax(v, "d"))
    val sum = result.data.sum
    assertEqualsDouble(sum, 1.0, 1e-10)
  }

  test("softmax forward: known values") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val result = EinEval.eval(Ein.Softmax(v, "d"))

    // Manual: exp(1) + exp(2) + exp(3) = e + e^2 + e^3
    val e1 = math.exp(1.0)
    val e2 = math.exp(2.0)
    val e3 = math.exp(3.0)
    val total = e1 + e2 + e3
    assertEqualsDouble(result.data(0), e1 / total, 1e-10)
    assertEqualsDouble(result.data(1), e2 / total, 1e-10)
    assertEqualsDouble(result.data(2), e3 / total, 1e-10)
  }

  test("softmax along specific dimension of 2D tensor") {
    val rows = Dim("i", 2)
    val cols = Dim("j", 3)
    val m = Ein.Param("M", List(rows, cols),
      TensorData.fromArray(List(rows, cols), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))

    // Softmax along "j" — each row sums to 1
    val result = EinEval.eval(Ein.Softmax(m, "j"))
    assertEquals(result.dims, List(rows, cols))

    // Row 0: softmax([1,2,3]) sums to 1
    val row0sum = result.data(0) + result.data(1) + result.data(2)
    assertEqualsDouble(row0sum, 1.0, 1e-10)

    // Row 1: softmax([4,5,6]) sums to 1
    val row1sum = result.data(3) + result.data(4) + result.data(5)
    assertEqualsDouble(row1sum, 1.0, 1e-10)
  }

  test("softmax numerical stability with large values") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1000.0, 1001.0, 1002.0)))
    val result = EinEval.eval(Ein.Softmax(v, "d"))

    // Should not produce NaN or Infinity
    result.data.foreach { x =>
      assert(!x.isNaN, s"Softmax produced NaN")
      assert(!x.isInfinite, s"Softmax produced Infinity")
    }
    // Should still sum to 1
    assertEqualsDouble(result.data.sum, 1.0, 1e-10)

    // With max-subtraction trick, equivalent to softmax([0, 1, 2])
    val e0 = math.exp(0.0)
    val e1 = math.exp(1.0)
    val e2 = math.exp(2.0)
    val total = e0 + e1 + e2
    assertEqualsDouble(result.data(0), e0 / total, 1e-10)
    assertEqualsDouble(result.data(1), e1 / total, 1e-10)
    assertEqualsDouble(result.data(2), e2 / total, 1e-10)
  }

  // --- LogSoftmax forward tests ---

  test("log_softmax equals log(softmax)") {
    val d = Dim("d", 4)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1.0, -0.5, 2.0, 0.3)))

    val softmaxResult = EinEval.eval(Ein.Softmax(v, "d"))
    val logSoftmaxResult = EinEval.eval(Ein.LogSoftmax(v, "d"))

    for i <- 0 until 4 do
      assertEqualsDouble(logSoftmaxResult.data(i), math.log(softmaxResult.data(i)), 1e-10,
        s"log_softmax[$i] != log(softmax[$i])")
  }

  test("log_softmax numerical stability with large values") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1000.0, 1001.0, 1002.0)))
    val result = EinEval.eval(Ein.LogSoftmax(v, "d"))

    result.data.foreach { x =>
      assert(!x.isNaN, s"LogSoftmax produced NaN")
      assert(!x.isInfinite, s"LogSoftmax produced Infinity")
    }

    // exp(log_softmax) should sum to 1
    val expSum = result.data.map(math.exp).sum
    assertEqualsDouble(expSum, 1.0, 1e-10)
  }

  // --- Gradient tests ---

  test("softmax gradient matches numerical (1D)") {
    val d = Dim("d", 4)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0)))
    val expr = Ein.Softmax(v, "d")
    verifyGrad(expr, "v", 4)
  }

  test("log_softmax gradient matches numerical (1D)") {
    val d = Dim("d", 4)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1.0, -0.5, 2.0, 0.3)))
    val expr = Ein.LogSoftmax(v, "d")
    verifyGrad(expr, "v", 4)
  }

  test("softmax gradient matches numerical (2D, along one dim)") {
    val rows = Dim("i", 2)
    val cols = Dim("j", 3)
    val m = Ein.Param("M", List(rows, cols),
      TensorData.fromArray(List(rows, cols), Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    val expr = Ein.Softmax(m, "j")
    verifyGrad(expr, "M", 6)
  }

  test("log_softmax gradient matches numerical (2D)") {
    val rows = Dim("i", 2)
    val cols = Dim("j", 3)
    val m = Ein.Param("M", List(rows, cols),
      TensorData.fromArray(List(rows, cols), Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    val expr = Ein.LogSoftmax(m, "j")
    verifyGrad(expr, "M", 6)
  }

  test("softmax gradient in dense layer: softmax(W*x + b)") {
    val features = Dim("j", 2)
    val classes = Dim("c", 3)
    val w = Ein.Param("W", List(classes, features),
      TensorData.fromArray(List(classes, features), Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    val b = Ein.Param("b", List(classes),
      TensorData.fromArray(List(classes), Array(0.1, -0.1, 0.0)))
    val x = Ein.Param("x", List(features),
      TensorData.fromArray(List(features), Array(1.0, -1.0)))
    val expr = Ein.Softmax((w × x) + b, "c")
    verifyGrad(expr, "W", 6)
    verifyGrad(expr, "b", 3)
    verifyGrad(expr, "x", 2)
  }

  // --- DSL and Block tests ---

  test("EinDsl softmax helper works") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val result1 = EinEval.eval(EinDsl.softmax(v, "d"))
    val result2 = EinEval.eval(Ein.Softmax(v, "d"))
    for i <- 0 until 3 do
      assertEqualsDouble(result1.data(i), result2.data(i), 1e-10)
  }

  test("Block.softmax materializes correctly") {
    val inp = Dim("inp", 3)
    val input = Ein.Input[Double]("x", List(inp))
    val block = Block.softmax[Double]("inp")
    val net = block.materialize(input)

    net match
      case Ein.Softmax(Ein.Input("x", _), "inp") => ()
      case other => fail(s"Expected Softmax(Input, 'inp'), got $other")
  }

  test("Block.softmax in pipeline: dense >> softmax") {
    val inp = Dim("inp", 2)
    val input = Ein.Input[Double]("x", List(inp))
    val net = (Block.dense[Double](3, Activation.ReLU) >> Block.softmax[Double]("block0_out"))
      .materialize(input)

    // Should be Softmax(Activate(ReLU, ...), "block0_out")
    net match
      case Ein.Softmax(Ein.Activate(Activation.ReLU, _), "block0_out") => ()
      case other => fail(s"Expected Softmax(ReLU(...), ...), got $other")
  }

  // --- Trace and Mermaid ---

  test("EinTrace handles softmax") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val expr = Ein.Softmax(v, "d")
    val graph = EinTrace.forward(expr)

    assertEquals(graph.nodes.size, 2) // Param + Softmax
    graph.nodes(graph.rootId).kind match
      case EinNodeKind.Softmax("d") => ()
      case other => fail(s"Expected Softmax node kind, got $other")
  }

  test("EinMermaid renders softmax") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val expr = Ein.Softmax(v, "d")
    val graph = EinTrace.forward(expr)
    val mermaid = EinMermaid.renderOps(graph)

    assert(mermaid.contains("softmax(d)"), s"Mermaid should contain 'softmax(d)': $mermaid")
  }
