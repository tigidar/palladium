package palladium.ein

class ReshapeSuite extends munit.FunSuite:

  val tolerance = 1e-9
  val gradTolerance = 1e-5
  val epsilon = 1e-5

  // ─── Forward eval ───

  test("reshape 1D to 2D: [6] -> [2, 3]") {
    val d = Dim("d", 6)
    val r = Dim("rows", 2)
    val c = Dim("cols", 3)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val expr = Ein.Reshape(v, List(r, c))

    assertEquals(expr.outputDims, List(r, c))
    val result = EinEval.eval(expr)
    assertEquals(result.dims, List(r, c))
    // Row-major: [[1,2,3],[4,5,6]]
    assertEqualsDouble(result.data(0), 1.0, tolerance)
    assertEqualsDouble(result.data(1), 2.0, tolerance)
    assertEqualsDouble(result.data(2), 3.0, tolerance)
    assertEqualsDouble(result.data(3), 4.0, tolerance)
    assertEqualsDouble(result.data(4), 5.0, tolerance)
    assertEqualsDouble(result.data(5), 6.0, tolerance)
  }

  test("reshape 2D to 1D: [2, 3] -> [6]") {
    val r = Dim("rows", 2)
    val c = Dim("cols", 3)
    val flat = Dim("flat", 6)
    val m = Ein.Param("m", List(r, c),
      TensorData.fromArray(List(r, c), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val expr = Ein.Reshape(m, List(flat))

    assertEquals(expr.outputDims, List(flat))
    val result = EinEval.eval(expr)
    assertEquals(result.dims, List(flat))
    for i <- 0 until 6 do
      assertEqualsDouble(result.data(i), (i + 1).toDouble, tolerance)
  }

  test("reshape split dim for multi-head attention: [seq, embed] -> [seq, heads, head_dim]") {
    val seq = Dim("seq", 3)
    val embed = Dim("embed", 4)
    val heads = Dim("heads", 2)
    val headDim = Dim("head_dim", 2)
    // 3x4 matrix, 12 elements
    val data = Array.tabulate(12)(i => (i + 1).toDouble)
    val m = Ein.Param("m", List(seq, embed),
      TensorData.fromArray(List(seq, embed), data))
    val expr = Ein.Reshape(m, List(seq, heads, headDim))

    assertEquals(expr.outputDims, List(seq, heads, headDim))
    val result = EinEval.eval(expr)
    assertEquals(result.dims, List(seq, heads, headDim))
    // Data layout unchanged (row-major reinterpretation)
    for i <- 0 until 12 do
      assertEqualsDouble(result.data(i), data(i), tolerance)
  }

  test("reshape merge dims: [seq, heads, head_dim] -> [seq, embed]") {
    val seq = Dim("seq", 3)
    val heads = Dim("heads", 2)
    val headDim = Dim("head_dim", 2)
    val embed = Dim("embed", 4)
    val data = Array.tabulate(12)(i => (i + 1).toDouble)
    val m = Ein.Param("m", List(seq, heads, headDim),
      TensorData.fromArray(List(seq, heads, headDim), data))
    val expr = Ein.Reshape(m, List(seq, embed))

    assertEquals(expr.outputDims, List(seq, embed))
    val result = EinEval.eval(expr)
    for i <- 0 until 12 do
      assertEqualsDouble(result.data(i), data(i), tolerance)
  }

  test("reshape preserves data through round-trip") {
    val a = Dim("a", 2)
    val b = Dim("b", 3)
    val c = Dim("c", 6)
    val data = Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    val m = Ein.Param("m", List(a, b),
      TensorData.fromArray(List(a, b), data))
    // [2,3] -> [6] -> [3,2]
    val flat = Ein.Reshape(m, List(c))
    val d1 = Dim("d1", 3)
    val d2 = Dim("d2", 2)
    val reshaped = Ein.Reshape(flat, List(d1, d2))
    val result = EinEval.eval(reshaped)
    assertEquals(result.dims, List(d1, d2))
    for i <- 0 until 6 do
      assertEqualsDouble(result.data(i), data(i), tolerance)
  }

  // ─── Backward (gradient) ───

  test("reshape gradient: 1D to 2D") {
    val d = Dim("d", 6)
    val r = Dim("rows", 2)
    val c = Dim("cols", 3)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val expr = Ein.Reshape(v, List(r, c))
    verifyGrad(expr, "v", 6)
  }

  test("reshape gradient: 2D to 3D (multi-head split)") {
    val seq = Dim("seq", 3)
    val embed = Dim("embed", 4)
    val heads = Dim("heads", 2)
    val headDim = Dim("head_dim", 2)
    val data = Array.tabulate(12)(i => (i + 1).toDouble)
    val m = Ein.Param("m", List(seq, embed),
      TensorData.fromArray(List(seq, embed), data))
    val expr = Ein.Reshape(m, List(seq, heads, headDim))
    verifyGrad(expr, "m", 12)
  }

  test("reshape gradient through downstream operations") {
    val d = Dim("d", 6)
    val r = Dim("rows", 2)
    val c = Dim("cols", 3)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    val reshaped = Ein.Reshape(v, List(r, c))
    val expr = Ein.Activate(Activation.Tanh, reshaped)
    verifyGrad(expr, "v", 6)
  }

  // ─── Trace ───

  test("reshape trace produces correct node") {
    val d = Dim("d", 6)
    val r = Dim("rows", 2)
    val c = Dim("cols", 3)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val expr = Ein.Reshape(v, List(r, c))
    val graph = EinTrace.forward(expr)
    assertEquals(graph.nodes.size, 2)
    val reshapeNode = graph.nodes(graph.rootId)
    assert(reshapeNode.kind.isInstanceOf[EinNodeKind.Reshape])
    assertEquals(reshapeNode.outputDims, List(r, c))
  }

  // ─── Mermaid ───

  test("reshape mermaid renders correctly") {
    val d = Dim("d", 6)
    val r = Dim("rows", 2)
    val c = Dim("cols", 3)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val expr = Ein.Reshape(v, List(r, c))
    val graph = EinTrace.forward(expr)
    val mermaid = EinMermaid.renderOps(graph)
    assert(mermaid.contains("reshape"), s"Expected 'reshape' in:\n$mermaid")
  }

  // ─── Symbolic gradient ───

  test("reshape symbolic gradient is inverse reshape") {
    val d = Dim("d", 6)
    val r = Dim("rows", 2)
    val c = Dim("cols", 3)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val expr = Ein.Reshape(v, List(r, c))
    val symGrads = EinSymbolicGrad.backward(expr)
    assert(symGrads.contains("v"))
    symGrads("v") match
      case Ein.Reshape(_, targetDims) =>
        assertEquals(targetDims, List(d)) // reshape back to original
      case other => fail(s"Expected Reshape node, got: $other")
  }

  // ─── Helpers ───

  def numericalGrad(
      expr: Ein[Double],
      paramId: String,
      elementIndex: Int
  ): Double =
    val plusExpr = perturbParam(expr, paramId, elementIndex, epsilon)
    val plusVal = EinEval.eval(plusExpr)
    val plusLoss = plusVal.data.sum

    val minusExpr = perturbParam(expr, paramId, elementIndex, -epsilon)
    val minusVal = EinEval.eval(minusExpr)
    val minusLoss = minusVal.data.sum

    (plusLoss - minusLoss) / (2 * epsilon)

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
      case Ein.Param(_, _, _) => expr
      case Ein.Input(_, _)    => expr
      case Ein.Fill(_, _)     => expr
      case Ein.Contract(l, r) =>
        Ein.Contract(perturbParam(l, paramId, elementIndex, delta),
                     perturbParam(r, paramId, elementIndex, delta))
      case Ein.ElemAdd(l, r) =>
        Ein.ElemAdd(perturbParam(l, paramId, elementIndex, delta),
                    perturbParam(r, paramId, elementIndex, delta))
      case Ein.ElemSub(l, r) =>
        Ein.ElemSub(perturbParam(l, paramId, elementIndex, delta),
                    perturbParam(r, paramId, elementIndex, delta))
      case Ein.ElemMul(l, r) =>
        Ein.ElemMul(perturbParam(l, paramId, elementIndex, delta),
                    perturbParam(r, paramId, elementIndex, delta))
      case Ein.Activate(f, arg) =>
        Ein.Activate(f, perturbParam(arg, paramId, elementIndex, delta))
      case Ein.ActivateDeriv(f, arg) =>
        Ein.ActivateDeriv(f, perturbParam(arg, paramId, elementIndex, delta))
      case Ein.ReduceSum(arg, over) =>
        Ein.ReduceSum(perturbParam(arg, paramId, elementIndex, delta), over)
      case Ein.Broadcast(arg, td) =>
        Ein.Broadcast(perturbParam(arg, paramId, elementIndex, delta), td)
      case Ein.Transpose(arg, perm) =>
        Ein.Transpose(perturbParam(arg, paramId, elementIndex, delta), perm)
      case Ein.Softmax(arg, overDim) =>
        Ein.Softmax(perturbParam(arg, paramId, elementIndex, delta), overDim)
      case Ein.LogSoftmax(arg, overDim) =>
        Ein.LogSoftmax(perturbParam(arg, paramId, elementIndex, delta), overDim)
      case Ein.LayerNorm(arg, s, b, od, e) =>
        Ein.LayerNorm(perturbParam(arg, paramId, elementIndex, delta),
          perturbParam(s, paramId, elementIndex, delta),
          perturbParam(b, paramId, elementIndex, delta), od, e)
      case Ein.Gather(table, indices, lookupDim) =>
        Ein.Gather(perturbParam(table, paramId, elementIndex, delta), indices, lookupDim)
      case Ein.Scatter(src, indices, lookupDim, tableDims) =>
        Ein.Scatter(perturbParam(src, paramId, elementIndex, delta), indices, lookupDim, tableDims)
      case Ein.Reshape(arg, targetDims) =>
        Ein.Reshape(perturbParam(arg, paramId, elementIndex, delta), targetDims)

  def verifyGrad(
      expr: Ein[Double],
      paramId: String,
      paramSize: Int
  ): Unit =
    val loss = Ein.ReduceSum(expr, expr.outputDims.map(_.name))
    val grads = EinGrad.backward(loss)
    val grad = grads(paramId)
    for idx <- 0 until paramSize do
      val analytical = grad.data(idx)
      val numerical = numericalGrad(loss, paramId, idx)
      assertEqualsDouble(
        analytical, numerical, gradTolerance,
        s"Gradient mismatch for $paramId[$idx]: analytical=$analytical, numerical=$numerical"
      )
