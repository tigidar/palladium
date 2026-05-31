package palladium.ein

class SliceSuite extends munit.FunSuite:

  val tolerance = 1e-9
  val gradTolerance = 1e-5
  val epsilon = 1e-5

  // ─── Forward eval ───

  test("slice 1D: take elements [1,3) from [5]") {
    val d = Dim("d", 5)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(10.0, 20.0, 30.0, 40.0, 50.0)))
    // Slice dim "d" from index 1 to 3 → new dim size 2
    val expr = Ein.Slice(v, "d", 1, 3)

    val sliced = Dim("d", 2)
    assertEquals(expr.outputDims, List(sliced))
    val result = EinEval.eval(expr)
    assertEquals(result.dims, List(sliced))
    assertEqualsDouble(result.data(0), 20.0, tolerance)
    assertEqualsDouble(result.data(1), 30.0, tolerance)
  }

  test("slice 2D: slice rows from matrix") {
    val rows = Dim("rows", 4)
    val cols = Dim("cols", 3)
    // [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    val data = Array.tabulate(12)(i => (i + 1).toDouble)
    val m = Ein.Param("m", List(rows, cols),
      TensorData.fromArray(List(rows, cols), data))
    // Slice rows [1,3) → rows 1 and 2
    val expr = Ein.Slice(m, "rows", 1, 3)

    val slicedRows = Dim("rows", 2)
    assertEquals(expr.outputDims, List(slicedRows, cols))
    val result = EinEval.eval(expr)
    assertEquals(result.dims, List(slicedRows, cols))
    // Row 1: [4,5,6], Row 2: [7,8,9]
    assertEqualsDouble(result.data(0), 4.0, tolerance)
    assertEqualsDouble(result.data(1), 5.0, tolerance)
    assertEqualsDouble(result.data(2), 6.0, tolerance)
    assertEqualsDouble(result.data(3), 7.0, tolerance)
    assertEqualsDouble(result.data(4), 8.0, tolerance)
    assertEqualsDouble(result.data(5), 9.0, tolerance)
  }

  test("slice 2D: slice columns from matrix") {
    val rows = Dim("rows", 2)
    val cols = Dim("cols", 4)
    // [[1,2,3,4],[5,6,7,8]]
    val data = Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    val m = Ein.Param("m", List(rows, cols),
      TensorData.fromArray(List(rows, cols), data))
    // Slice cols [1,3) → cols 1 and 2
    val expr = Ein.Slice(m, "cols", 1, 3)

    val slicedCols = Dim("cols", 2)
    assertEquals(expr.outputDims, List(rows, slicedCols))
    val result = EinEval.eval(expr)
    assertEquals(result.dims, List(rows, slicedCols))
    // [2,3], [6,7]
    assertEqualsDouble(result.data(0), 2.0, tolerance)
    assertEqualsDouble(result.data(1), 3.0, tolerance)
    assertEqualsDouble(result.data(2), 6.0, tolerance)
    assertEqualsDouble(result.data(3), 7.0, tolerance)
  }

  test("slice full range is identity") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val expr = Ein.Slice(v, "d", 0, 3)
    val result = EinEval.eval(expr)
    assertEquals(result.dims, List(d))
    for i <- 0 until 3 do
      assertEqualsDouble(result.data(i), (i + 1).toDouble, tolerance)
  }

  test("slice single element") {
    val d = Dim("d", 5)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(10.0, 20.0, 30.0, 40.0, 50.0)))
    val expr = Ein.Slice(v, "d", 2, 3)
    val sliced = Dim("d", 1)
    assertEquals(expr.outputDims, List(sliced))
    val result = EinEval.eval(expr)
    assertEqualsDouble(result.data(0), 30.0, tolerance)
  }

  test("slice 3D tensor along middle dim") {
    val a = Dim("a", 2)
    val b = Dim("b", 4)
    val c = Dim("c", 3)
    val data = Array.tabulate(24)(i => i.toDouble)
    val t = Ein.Param("t", List(a, b, c),
      TensorData.fromArray(List(a, b, c), data))
    // Slice b [1,3) → keep b indices 1 and 2
    val expr = Ein.Slice(t, "b", 1, 3)
    val slicedB = Dim("b", 2)
    assertEquals(expr.outputDims, List(a, slicedB, c))
    val result = EinEval.eval(expr)
    assertEquals(result.dims, List(a, slicedB, c))
    // a=0, b=1: offset 1*3=3 → [3,4,5]
    // a=0, b=2: offset 2*3=6 → [6,7,8]
    // a=1, b=1: offset 12+3=15 → [15,16,17] (wait, a=1 starts at 12 since 4*3=12 per a)
    // a=1, b=2: offset 12+6=18 → [18,19,20]
    assertEqualsDouble(result.data(0), 3.0, tolerance)
    assertEqualsDouble(result.data(1), 4.0, tolerance)
    assertEqualsDouble(result.data(2), 5.0, tolerance)
    assertEqualsDouble(result.data(3), 6.0, tolerance)
    assertEqualsDouble(result.data(4), 7.0, tolerance)
    assertEqualsDouble(result.data(5), 8.0, tolerance)
    assertEqualsDouble(result.data(6), 15.0, tolerance)
    assertEqualsDouble(result.data(7), 16.0, tolerance)
    assertEqualsDouble(result.data(8), 17.0, tolerance)
    assertEqualsDouble(result.data(9), 18.0, tolerance)
    assertEqualsDouble(result.data(10), 19.0, tolerance)
    assertEqualsDouble(result.data(11), 20.0, tolerance)
  }

  // ─── Backward (gradient) ───

  test("slice gradient: 1D") {
    val d = Dim("d", 5)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0, 5.0)))
    val expr = Ein.Slice(v, "d", 1, 3)
    verifyGrad(expr, "v", 5)
  }

  test("slice gradient: 2D slice rows") {
    val rows = Dim("rows", 4)
    val cols = Dim("cols", 3)
    val data = Array.tabulate(12)(i => (i + 1).toDouble)
    val m = Ein.Param("m", List(rows, cols),
      TensorData.fromArray(List(rows, cols), data))
    val expr = Ein.Slice(m, "rows", 1, 3)
    verifyGrad(expr, "m", 12)
  }

  test("slice gradient: 2D slice cols") {
    val rows = Dim("rows", 2)
    val cols = Dim("cols", 4)
    val data = Array.tabulate(8)(i => (i + 1).toDouble)
    val m = Ein.Param("m", List(rows, cols),
      TensorData.fromArray(List(rows, cols), data))
    val expr = Ein.Slice(m, "cols", 1, 3)
    verifyGrad(expr, "m", 8)
  }

  test("slice gradient through downstream operations") {
    val d = Dim("d", 5)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(0.1, 0.2, 0.3, 0.4, 0.5)))
    val sliced = Ein.Slice(v, "d", 1, 4)
    val expr = Ein.Activate(Activation.Tanh, sliced)
    verifyGrad(expr, "v", 5)
  }

  // ─── Trace ───

  test("slice trace produces correct node") {
    val d = Dim("d", 5)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0, 5.0)))
    val expr = Ein.Slice(v, "d", 1, 3)
    val graph = EinTrace.forward(expr)
    assertEquals(graph.nodes.size, 2)
    val sliceNode = graph.nodes(graph.rootId)
    assert(sliceNode.kind.isInstanceOf[EinNodeKind.Slice])
    assertEquals(sliceNode.outputDims, List(Dim("d", 2)))
  }

  // ─── Mermaid ───

  test("slice mermaid renders correctly") {
    val d = Dim("d", 5)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0, 5.0)))
    val expr = Ein.Slice(v, "d", 1, 3)
    val graph = EinTrace.forward(expr)
    val mermaid = EinMermaid.renderOps(graph)
    assert(mermaid.contains("slice"), s"Expected 'slice' in:\n$mermaid")
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
      case Ein.Reshape(arg, targetDims) =>
        Ein.Reshape(perturbParam(arg, paramId, elementIndex, delta), targetDims)
      case Ein.Gather(table, indices, lookupDim) =>
        Ein.Gather(perturbParam(table, paramId, elementIndex, delta), indices, lookupDim)
      case Ein.Scatter(src, indices, lookupDim, tableDims) =>
        Ein.Scatter(perturbParam(src, paramId, elementIndex, delta), indices, lookupDim, tableDims)
      case Ein.Slice(arg, dim, from, to) =>
        Ein.Slice(perturbParam(arg, paramId, elementIndex, delta), dim, from, to)

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
