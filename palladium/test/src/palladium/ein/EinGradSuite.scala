package palladium.ein

import palladium.NumberLike

class EinGradSuite extends munit.FunSuite:

  val tolerance = 1e-5
  val epsilon = 1e-5

  /** Numerical gradient via central finite differences for a single element of a Param tensor */
  def numericalGrad(
      expr: Ein[Double],
      paramId: String,
      elementIndex: Int,
      feed: Map[String, TensorData[Double]] = Map.empty
  ): Double =
    // Perturb +epsilon
    val plusExpr = perturbParam(expr, paramId, elementIndex, epsilon)
    val plusVal = EinEval.eval(plusExpr, feed)
    val plusLoss = plusVal.data.sum

    // Perturb -epsilon
    val minusExpr = perturbParam(expr, paramId, elementIndex, -epsilon)
    val minusVal = EinEval.eval(minusExpr, feed)
    val minusLoss = minusVal.data.sum

    (plusLoss - minusLoss) / (2 * epsilon)

  /** Create a copy of the expression with one element of a Param perturbed */
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
        Ein.LayerNorm(perturbParam(arg, paramId, elementIndex, delta), perturbParam(s, paramId, elementIndex, delta), perturbParam(b, paramId, elementIndex, delta), od, e)
      case Ein.Reshape(arg, targetDims) =>
        Ein.Reshape(perturbParam(arg, paramId, elementIndex, delta), targetDims)
      case Ein.Slice(arg, dim, from, to) =>
        Ein.Slice(perturbParam(arg, paramId, elementIndex, delta), dim, from, to)
      case Ein.Gather(table, indices, lookupDim) =>
        Ein.Gather(perturbParam(table, paramId, elementIndex, delta), indices, lookupDim)
      case Ein.Scatter(src, indices, lookupDim, tableDims) =>
        Ein.Scatter(perturbParam(src, paramId, elementIndex, delta), indices, lookupDim, tableDims)

  /** Verify all elements of a param's gradient match numerical */
  def verifyGrad(
      expr: Ein[Double],
      paramId: String,
      paramSize: Int,
      feed: Map[String, TensorData[Double]] = Map.empty
  ): Unit =
    // To get scalar loss, wrap in ReduceSum to get sum of all elements
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

  test("gradient of dot product") {
    val d = Dim("d", 3)
    val v1 = Ein.Param("v1", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val v2 = Ein.Param("v2", List(d), TensorData.fromArray(List(d), Array(4.0, 5.0, 6.0)))
    val dot = v1 × v2 // scalar output, so sum is identity
    val grads = EinGrad.backward(dot)
    // d(v1.v2)/dv1 = v2, d(v1.v2)/dv2 = v1
    val g1 = grads("v1")
    val g2 = grads("v2")
    assertEqualsDouble(g1.data(0), 4.0, tolerance)
    assertEqualsDouble(g1.data(1), 5.0, tolerance)
    assertEqualsDouble(g1.data(2), 6.0, tolerance)
    assertEqualsDouble(g2.data(0), 1.0, tolerance)
    assertEqualsDouble(g2.data(1), 2.0, tolerance)
    assertEqualsDouble(g2.data(2), 3.0, tolerance)
  }

  test("gradient of matrix-vector multiply matches numerical") {
    val rows = Dim("i", 2)
    val cols = Dim("j", 3)
    val m = Ein.Param("M", List(rows, cols),
      TensorData.fromArray(List(rows, cols), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val v = Ein.Param("v", List(cols),
      TensorData.fromArray(List(cols), Array(0.5, -0.5, 1.0)))
    val expr = m × v
    verifyGrad(expr, "M", 6)
    verifyGrad(expr, "v", 3)
  }

  test("gradient of element-wise add") {
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val b = Ein.Param("b", List(d), TensorData.fromArray(List(d), Array(4.0, 5.0, 6.0)))
    val expr = a + b
    verifyGrad(expr, "a", 3)
    verifyGrad(expr, "b", 3)
  }

  test("gradient of element-wise sub") {
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val b = Ein.Param("b", List(d), TensorData.fromArray(List(d), Array(4.0, 5.0, 6.0)))
    val expr = a - b
    verifyGrad(expr, "a", 3)
    verifyGrad(expr, "b", 3)
  }

  test("gradient of element-wise multiply") {
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(2.0, 3.0, 4.0)))
    val b = Ein.Param("b", List(d), TensorData.fromArray(List(d), Array(5.0, 6.0, 7.0)))
    val expr = a.elemMul(b)
    verifyGrad(expr, "a", 3)
    verifyGrad(expr, "b", 3)
  }

  test("gradient of relu") {
    val d = Dim("d", 4)
    // Avoid 0 in input for finite difference stability
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(-2.0, -0.5, 0.5, 3.0)))
    val expr = Ein.Activate(Activation.ReLU, v)
    verifyGrad(expr, "v", 4)
  }

  test("gradient of sigmoid") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(-1.0, 0.0, 1.0)))
    val expr = Ein.Activate(Activation.Sigmoid, v)
    verifyGrad(expr, "v", 3)
  }

  test("gradient of tanh") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(-1.0, 0.0, 1.0)))
    val expr = Ein.Activate(Activation.Tanh, v)
    verifyGrad(expr, "v", 3)
  }

  test("gradient of matrix-matrix multiply matches numerical") {
    val mi = Dim("i", 2)
    val mk = Dim("k", 3)
    val mj = Dim("j", 2)
    val a = Ein.Param("A", List(mi, mk),
      TensorData.fromArray(List(mi, mk), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val b = Ein.Param("B", List(mk, mj),
      TensorData.fromArray(List(mk, mj), Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    val expr = a × b
    verifyGrad(expr, "A", 6)
    verifyGrad(expr, "B", 6)
  }

  test("gradient of dense layer: relu(W*x + b)") {
    val features = Dim("j", 3)
    val hidden = Dim("i", 2)
    val w = Ein.Param("W", List(hidden, features),
      TensorData.fromArray(List(hidden, features), Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    val b = Ein.Param("b", List(hidden),
      TensorData.fromArray(List(hidden), Array(0.1, -0.1)))
    val x = Ein.Param("x", List(features),
      TensorData.fromArray(List(features), Array(1.0, 2.0, 3.0)))
    val expr = Ein.Activate(Activation.ReLU, (w × x) + b)
    verifyGrad(expr, "W", 6)
    verifyGrad(expr, "b", 2)
    verifyGrad(expr, "x", 3)
  }

  test("gradient of reduce sum") {
    val di = Dim("i", 2)
    val dj = Dim("j", 3)
    val m = Ein.Param("M", List(di, dj),
      TensorData.fromArray(List(di, dj), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val expr = Ein.ReduceSum(m, List("j"))
    verifyGrad(expr, "M", 6)
  }

  test("gradient of broadcast") {
    val di = Dim("i", 3)
    val v = Ein.Param("v", Nil, TensorData.scalar(2.0))
    val expr = Ein.Broadcast(v, List(di))
    verifyGrad(expr, "v", 1)
  }

  test("gradient of transpose") {
    val di = Dim("i", 2)
    val dj = Dim("j", 3)
    val m = Ein.Param("M", List(di, dj),
      TensorData.fromArray(List(di, dj), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val expr = Ein.Transpose(m, List("j", "i"))
    verifyGrad(expr, "M", 6)
  }

  test("gradient of broadcasting add (bias + matrix)") {
    val di = Dim("i", 2)
    val dj = Dim("j", 3)
    val m = Ein.Param("M", List(di, dj),
      TensorData.fromArray(List(di, dj), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val b = Ein.Param("b", List(di),
      TensorData.fromArray(List(di), Array(0.1, 0.2)))
    val expr = m + b
    verifyGrad(expr, "M", 6)
    verifyGrad(expr, "b", 2)
  }

  test("end-to-end: 2-layer network gradients match numerical") {
    val inp = Dim("inp", 2)
    val hid = Dim("hid", 3)
    val out = Dim("out", 1)

    val w1 = Ein.Param("W1", List(hid, inp),
      TensorData.fromArray(List(hid, inp), Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    val b1 = Ein.Param("b1", List(hid),
      TensorData.fromArray(List(hid), Array(0.0, 0.0, 0.0)))
    val w2 = Ein.Param("W2", List(out, hid),
      TensorData.fromArray(List(out, hid), Array(0.1, 0.2, 0.3)))
    val b2 = Ein.Param("b2", List(out),
      TensorData.fromArray(List(out), Array(0.0)))
    val x = Ein.Param("x", List(inp),
      TensorData.fromArray(List(inp), Array(1.0, -1.0)))

    // h = relu(W1*x + b1), y = W2*h + b2
    val h = Ein.Activate(Activation.ReLU, (w1 × x) + b1)
    val y = (w2 × h) + b2

    verifyGrad(y, "W1", 6)
    verifyGrad(y, "b1", 3)
    verifyGrad(y, "W2", 3)
    verifyGrad(y, "b2", 1)
    verifyGrad(y, "x", 2)
  }
