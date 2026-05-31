package palladium.ein

import palladium.NumberLike

class EinSymbolicGradSuite extends munit.FunSuite:

  val tolerance = 1e-5

  /** Verify symbolic gradients match numerical (via EinGrad) for a given param */
  def verifySymbolicMatchesNumerical(
      expr: Ein[Double],
      paramId: String,
      paramSize: Int,
      feed: Map[String, TensorData[Double]] = Map.empty
  ): Unit =
    // Wrap in ReduceSum to get scalar loss
    val loss = Ein.ReduceSum(expr, expr.outputDims.map(_.name))
    val symbolicGrads = EinSymbolicGrad.backward(loss)
    val numericalGrads = EinGrad.backward(loss, feed)

    assert(symbolicGrads.contains(paramId), s"Missing symbolic gradient for $paramId")
    assert(numericalGrads.contains(paramId), s"Missing numerical gradient for $paramId")

    val symbolicGradExpr = symbolicGrads(paramId)
    val symbolicResult = EinEval.eval(symbolicGradExpr, feed)
    val numericalResult = numericalGrads(paramId)

    assertEquals(symbolicResult.dims.map(_.name), numericalResult.dims.map(_.name),
      s"Dimension name mismatch for $paramId gradient")

    for idx <- 0 until paramSize do
      assertEqualsDouble(
        symbolicResult.data(idx), numericalResult.data(idx), tolerance,
        s"Symbolic/numerical gradient mismatch for $paramId[$idx]: " +
          s"symbolic=${symbolicResult.data(idx)}, numerical=${numericalResult.data(idx)}"
      )

  test("symbolic gradient of element-wise add") {
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val b = Ein.Param("b", List(d), TensorData.fromArray(List(d), Array(4.0, 5.0, 6.0)))
    val expr = a + b
    verifySymbolicMatchesNumerical(expr, "a", 3)
    verifySymbolicMatchesNumerical(expr, "b", 3)
  }

  test("symbolic gradient of element-wise sub") {
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val b = Ein.Param("b", List(d), TensorData.fromArray(List(d), Array(4.0, 5.0, 6.0)))
    val expr = a - b
    verifySymbolicMatchesNumerical(expr, "a", 3)
    verifySymbolicMatchesNumerical(expr, "b", 3)
  }

  test("symbolic gradient of element-wise mul") {
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(2.0, 3.0, 4.0)))
    val b = Ein.Param("b", List(d), TensorData.fromArray(List(d), Array(5.0, 6.0, 7.0)))
    val expr = a.elemMul(b)
    verifySymbolicMatchesNumerical(expr, "a", 3)
    verifySymbolicMatchesNumerical(expr, "b", 3)
  }

  test("symbolic gradient of contraction (dot product)") {
    val d = Dim("d", 3)
    val v1 = Ein.Param("v1", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val v2 = Ein.Param("v2", List(d), TensorData.fromArray(List(d), Array(4.0, 5.0, 6.0)))
    val expr = v1 * v2

    val grads = EinSymbolicGrad.backward(expr)
    assert(grads.contains("v1"))
    assert(grads.contains("v2"))

    // d(v1.v2)/dv1 = v2
    val g1 = EinEval.eval(grads("v1"))
    assertEqualsDouble(g1.data(0), 4.0, tolerance)
    assertEqualsDouble(g1.data(1), 5.0, tolerance)
    assertEqualsDouble(g1.data(2), 6.0, tolerance)
  }

  test("symbolic gradient of matrix-vector multiply") {
    val rows = Dim("i", 2)
    val cols = Dim("j", 3)
    val m = Ein.Param("M", List(rows, cols),
      TensorData.fromArray(List(rows, cols), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val v = Ein.Param("v", List(cols),
      TensorData.fromArray(List(cols), Array(0.5, -0.5, 1.0)))
    val expr = m * v
    verifySymbolicMatchesNumerical(expr, "M", 6)
    verifySymbolicMatchesNumerical(expr, "v", 3)
  }

  test("symbolic gradient of activation (relu)") {
    val d = Dim("d", 4)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(-2.0, -0.5, 0.5, 3.0)))
    val expr = Ein.Activate(Activation.ReLU, v)
    verifySymbolicMatchesNumerical(expr, "v", 4)
  }

  test("symbolic gradient of activation (sigmoid)") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(-1.0, 0.0, 1.0)))
    val expr = Ein.Activate(Activation.Sigmoid, v)
    verifySymbolicMatchesNumerical(expr, "v", 3)
  }

  test("symbolic gradient of activation (tanh)") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(-1.0, 0.0, 1.0)))
    val expr = Ein.Activate(Activation.Tanh, v)
    verifySymbolicMatchesNumerical(expr, "v", 3)
  }

  test("symbolic gradient of reduce sum") {
    val di = Dim("i", 2)
    val dj = Dim("j", 3)
    val m = Ein.Param("M", List(di, dj),
      TensorData.fromArray(List(di, dj), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val expr = Ein.ReduceSum(m, List("j"))
    verifySymbolicMatchesNumerical(expr, "M", 6)
  }

  test("symbolic gradient of broadcast") {
    val di = Dim("i", 3)
    val v = Ein.Param("v", Nil, TensorData.scalar(2.0))
    val expr = Ein.Broadcast(v, List(di))
    verifySymbolicMatchesNumerical(expr, "v", 1)
  }

  test("symbolic gradient of transpose") {
    val di = Dim("i", 2)
    val dj = Dim("j", 3)
    val m = Ein.Param("M", List(di, dj),
      TensorData.fromArray(List(di, dj), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val expr = Ein.Transpose(m, List("j", "i"))
    verifySymbolicMatchesNumerical(expr, "M", 6)
  }

  test("symbolic gradient of dense layer: relu(W*x + b)") {
    val features = Dim("j", 3)
    val hidden = Dim("i", 2)
    val w = Ein.Param("W", List(hidden, features),
      TensorData.fromArray(List(hidden, features), Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    val b = Ein.Param("b", List(hidden),
      TensorData.fromArray(List(hidden), Array(0.1, -0.1)))
    val x = Ein.Param("x", List(features),
      TensorData.fromArray(List(features), Array(1.0, 2.0, 3.0)))
    val expr = Ein.Activate(Activation.ReLU, (w * x) + b)
    verifySymbolicMatchesNumerical(expr, "W", 6)
    verifySymbolicMatchesNumerical(expr, "b", 2)
    verifySymbolicMatchesNumerical(expr, "x", 3)
  }

  test("symbolic gradient of reshape") {
    val di = Dim("i", 2)
    val dj = Dim("j", 3)
    val flat = Dim("flat", 6)
    val m = Ein.Param("M", List(di, dj),
      TensorData.fromArray(List(di, dj), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val expr = Ein.Reshape(m, List(flat))
    verifySymbolicMatchesNumerical(expr, "M", 6)
  }

  test("symbolic gradient of gather/scatter round-trip") {
    val vocab = Dim("vocab", 4)
    val embed = Dim("embed", 2)
    val table = Ein.Param("table", List(vocab, embed),
      TensorData.fromArray(List(vocab, embed), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)))
    val seq = Dim("seq", 3)
    val indices = TensorData.fromArray[Int](List(seq), Array(0, 2, 1))
    val expr = Ein.Gather(table, indices, "vocab")
    verifySymbolicMatchesNumerical(expr, "table", 8)
  }

  test("symbolic gradient skips non-param leaves") {
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val b = Ein.Input[Double]("b", List(d))
    val ones = Ein.Ones[Double](List(d))

    val expr = a + ones
    val loss = Ein.ReduceSum(expr, List("d"))
    val grads = EinSymbolicGrad.backward(loss)

    assert(grads.contains("a"), "Should have gradient for param 'a'")
    assert(!grads.contains("b"), "Should not have gradient for input 'b'")
  }

  test("symbolic gradient accumulates for shared params") {
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    // a + a should give gradient 2*ones
    val expr = a + a
    val loss = Ein.ReduceSum(expr, List("d"))
    val grads = EinSymbolicGrad.backward(loss)
    val gradResult = EinEval.eval(grads("a"))
    // Each element should be 2.0 (ones + ones)
    for i <- 0 until 3 do
      assertEqualsDouble(gradResult.data(i), 2.0, tolerance)
  }
