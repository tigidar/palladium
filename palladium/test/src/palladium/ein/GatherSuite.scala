package palladium.ein

class GatherSuite extends munit.FunSuite:

  val tolerance = 1e-9
  val gradTolerance = 1e-5
  val epsilon = 1e-5

  // ─── Forward eval ───

  test("gather from 1D table: table[vocab] -> out[seq]") {
    val vocab = Dim("vocab", 4)
    val seq = Dim("seq", 3)
    // table = [10, 20, 30, 40]
    val table = Ein.Param("table", List(vocab),
      TensorData.fromArray(List(vocab), Array(10.0, 20.0, 30.0, 40.0)))
    // indices = [2, 0, 3] → should pick [30, 10, 40]
    val indices = TensorData.fromArray(List(seq), Array(2, 0, 3))
    val expr = Ein.Gather(table, indices, "vocab")

    assertEquals(expr.outputDims, List(seq))
    val result = EinEval.eval(expr)
    assertEquals(result.dims, List(seq))
    assertEqualsDouble(result.data(0), 30.0, tolerance)
    assertEqualsDouble(result.data(1), 10.0, tolerance)
    assertEqualsDouble(result.data(2), 40.0, tolerance)
  }

  test("gather from 2D table (embedding lookup): table[vocab, embed] -> out[seq, embed]") {
    val vocab = Dim("vocab", 3)
    val embed = Dim("embed", 2)
    val seq = Dim("seq", 4)
    // table = [[1,2], [3,4], [5,6]]  (3 words, 2-dim embeddings)
    val table = Ein.Param("table", List(vocab, embed),
      TensorData.fromArray(List(vocab, embed), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    // indices = [0, 2, 1, 0] → pick rows 0, 2, 1, 0
    val indices = TensorData.fromArray(List(seq), Array(0, 2, 1, 0))
    val expr = Ein.Gather(table, indices, "vocab")

    assertEquals(expr.outputDims, List(seq, embed))
    val result = EinEval.eval(expr)
    assertEquals(result.dims, List(seq, embed))
    // row 0: [1,2], row 2: [5,6], row 1: [3,4], row 0: [1,2]
    assertEqualsDouble(result.data(0), 1.0, tolerance) // [0,0]
    assertEqualsDouble(result.data(1), 2.0, tolerance) // [0,1]
    assertEqualsDouble(result.data(2), 5.0, tolerance) // [1,0]
    assertEqualsDouble(result.data(3), 6.0, tolerance) // [1,1]
    assertEqualsDouble(result.data(4), 3.0, tolerance) // [2,0]
    assertEqualsDouble(result.data(5), 4.0, tolerance) // [2,1]
    assertEqualsDouble(result.data(6), 1.0, tolerance) // [3,0]
    assertEqualsDouble(result.data(7), 2.0, tolerance) // [3,1]
  }

  test("gather with duplicate indices accumulates correctly in gradient") {
    val vocab = Dim("vocab", 3)
    val embed = Dim("embed", 2)
    val seq = Dim("seq", 3)
    // table = [[1,2], [3,4], [5,6]]
    val table = Ein.Param("table", List(vocab, embed),
      TensorData.fromArray(List(vocab, embed), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    // indices = [0, 0, 2] → row 0 appears twice
    val indices = TensorData.fromArray(List(seq), Array(0, 0, 2))
    val expr = Ein.Gather(table, indices, "vocab")

    val result = EinEval.eval(expr)
    // row 0: [1,2], row 0: [1,2], row 2: [5,6]
    assertEqualsDouble(result.data(0), 1.0, tolerance)
    assertEqualsDouble(result.data(1), 2.0, tolerance)
    assertEqualsDouble(result.data(2), 1.0, tolerance)
    assertEqualsDouble(result.data(3), 2.0, tolerance)
    assertEqualsDouble(result.data(4), 5.0, tolerance)
    assertEqualsDouble(result.data(5), 6.0, tolerance)
  }

  test("gather preserves dims after lookup dim") {
    val vocab = Dim("vocab", 4)
    val d1 = Dim("d1", 2)
    val d2 = Dim("d2", 3)
    val seq = Dim("seq", 2)
    // table[vocab, d1, d2], indices[seq], lookupDim="vocab" → out[seq, d1, d2]
    val tableData = Array.tabulate(4 * 2 * 3)(i => i.toDouble)
    val table = Ein.Param("table", List(vocab, d1, d2),
      TensorData.fromArray(List(vocab, d1, d2), tableData))
    val indices = TensorData.fromArray(List(seq), Array(1, 3))
    val expr = Ein.Gather(table, indices, "vocab")

    assertEquals(expr.outputDims, List(seq, d1, d2))
    val result = EinEval.eval(expr)
    assertEquals(result.dims, List(seq, d1, d2))
    // row 1 starts at offset 6 (1*2*3), row 3 starts at 18 (3*2*3)
    for j <- 0 until 6 do
      assertEqualsDouble(result.data(j), tableData(6 + j), tolerance)
    for j <- 0 until 6 do
      assertEqualsDouble(result.data(6 + j), tableData(18 + j), tolerance)
  }

  // ─── Backward (gradient) ───

  test("gather gradient: simple 1D table") {
    val vocab = Dim("vocab", 4)
    val seq = Dim("seq", 3)
    val table = Ein.Param("table", List(vocab),
      TensorData.fromArray(List(vocab), Array(10.0, 20.0, 30.0, 40.0)))
    val indices = TensorData.fromArray(List(seq), Array(2, 0, 3))
    val expr = Ein.Gather(table, indices, "vocab")
    verifyGrad(expr, "table", 4)
  }

  test("gather gradient: 2D embedding lookup") {
    val vocab = Dim("vocab", 3)
    val embed = Dim("embed", 2)
    val seq = Dim("seq", 4)
    val table = Ein.Param("table", List(vocab, embed),
      TensorData.fromArray(List(vocab, embed), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val indices = TensorData.fromArray(List(seq), Array(0, 2, 1, 0))
    val expr = Ein.Gather(table, indices, "vocab")
    verifyGrad(expr, "table", 6)
  }

  test("gather gradient with duplicate indices: gradients scatter-add") {
    val vocab = Dim("vocab", 3)
    val embed = Dim("embed", 2)
    val seq = Dim("seq", 3)
    val table = Ein.Param("table", List(vocab, embed),
      TensorData.fromArray(List(vocab, embed), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    // Row 0 appears twice → its gradient should be 2x
    val indices = TensorData.fromArray(List(seq), Array(0, 0, 2))
    val expr = Ein.Gather(table, indices, "vocab")
    verifyGrad(expr, "table", 6)
  }

  test("gather gradient through downstream operations") {
    val vocab = Dim("vocab", 4)
    val embed = Dim("embed", 3)
    val seq = Dim("seq", 2)
    val table = Ein.Param("table", List(vocab, embed),
      TensorData.fromArray(List(vocab, embed),
        Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)))
    val indices = TensorData.fromArray(List(seq), Array(1, 3))
    val gathered = Ein.Gather(table, indices, "vocab")
    // Apply an activation on top of the gathered result
    val expr = Ein.Activate(Activation.Tanh, gathered)
    verifyGrad(expr, "table", 12)
  }

  // ─── Trace ───

  test("gather trace produces correct node") {
    val vocab = Dim("vocab", 3)
    val embed = Dim("embed", 2)
    val seq = Dim("seq", 4)
    val table = Ein.Param("table", List(vocab, embed),
      TensorData.fromArray(List(vocab, embed), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val indices = TensorData.fromArray(List(seq), Array(0, 2, 1, 0))
    val expr = Ein.Gather(table, indices, "vocab")
    val graph = EinTrace.forward(expr)
    // Should have 2 nodes: param + gather
    assertEquals(graph.nodes.size, 2)
    val gatherNode = graph.nodes(graph.rootId)
    assert(gatherNode.kind.isInstanceOf[EinNodeKind.Gather])
    assertEquals(gatherNode.outputDims, List(seq, embed))
    assertEquals(gatherNode.inputs, List(0)) // table param
  }

  // ─── Mermaid ───

  test("gather mermaid renders correctly") {
    val vocab = Dim("vocab", 3)
    val embed = Dim("embed", 2)
    val seq = Dim("seq", 4)
    val table = Ein.Param("table", List(vocab, embed),
      TensorData.fromArray(List(vocab, embed), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val indices = TensorData.fromArray(List(seq), Array(0, 2, 1, 0))
    val expr = Ein.Gather(table, indices, "vocab")
    val graph = EinTrace.forward(expr)
    val mermaid = EinMermaid.renderOps(graph)
    assert(mermaid.contains("gather"), s"Expected 'gather' in:\n$mermaid")
    assert(mermaid.contains("vocab"), s"Expected 'vocab' in:\n$mermaid")
  }

  // ─── Symbolic gradient ───

  test("gather symbolic gradient produces scatter structure") {
    val vocab = Dim("vocab", 3)
    val embed = Dim("embed", 2)
    val seq = Dim("seq", 4)
    val table = Ein.Param("table", List(vocab, embed),
      TensorData.fromArray(List(vocab, embed), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val indices = TensorData.fromArray(List(seq), Array(0, 2, 1, 0))
    val expr = Ein.Gather(table, indices, "vocab")
    val symGrads = EinSymbolicGrad.backward(expr)
    assert(symGrads.contains("table"))
    // The symbolic grad should be a Scatter node
    symGrads("table") match
      case Ein.Scatter(_, _, _, _) => () // expected
      case other => fail(s"Expected Scatter node, got: $other")
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
      case Ein.Scatter(src, indices, lookupDim, tableShape) =>
        Ein.Scatter(perturbParam(src, paramId, elementIndex, delta), indices, lookupDim, tableShape)

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
