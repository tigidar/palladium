package palladium.ein

import palladium.NumberLike

class ActivationSuite extends munit.FunSuite:

  val tolerance = 1e-6
  val fdEpsilon = 1e-7

  // --- GELU forward tests ---

  test("GELU(0) ≈ 0") {
    val result = Activation(Activation.GELU, 0.0)
    assertEqualsDouble(result, 0.0, tolerance)
  }

  test("GELU(x) ≈ x for large positive x") {
    // GELU(x) → x as x → ∞
    val result = Activation(Activation.GELU, 5.0)
    assertEqualsDouble(result, 5.0, 0.01)
  }

  test("GELU(x) ≈ 0 for large negative x") {
    // GELU(x) → 0 as x → -∞
    val result = Activation(Activation.GELU, -5.0)
    assertEqualsDouble(result, 0.0, 0.01)
  }

  test("GELU known values") {
    // GELU(1.0) ≈ 0.8412 (from PyTorch reference)
    val result = Activation(Activation.GELU, 1.0)
    assertEqualsDouble(result, 0.8412, 0.001)

    // GELU(-1.0) ≈ -0.1588
    val negResult = Activation(Activation.GELU, -1.0)
    assertEqualsDouble(negResult, -0.1588, 0.001)
  }

  test("GELU is monotonically increasing for x > -0.5") {
    val x1 = Activation(Activation.GELU, 0.0)
    val x2 = Activation(Activation.GELU, 0.5)
    val x3 = Activation(Activation.GELU, 1.0)
    assert(x1 < x2)
    assert(x2 < x3)
  }

  test("GELU derivative matches finite differences") {
    for x <- List(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0) do
      val analytical = Activation.deriv(Activation.GELU, x)
      val numerical = (Activation(Activation.GELU, x + fdEpsilon) - Activation(Activation.GELU, x - fdEpsilon)) / (2 * fdEpsilon)
      assertEqualsDouble(analytical, numerical, 1e-4,
        s"GELU deriv mismatch at x=$x: analytical=$analytical, numerical=$numerical")
  }

  // --- Swish forward tests ---

  test("Swish(0) = 0") {
    val result = Activation(Activation.Swish, 0.0)
    assertEqualsDouble(result, 0.0, tolerance)
  }

  test("Swish(x) = x * sigmoid(x)") {
    for x <- List(-2.0, -1.0, 0.5, 1.0, 2.0) do
      val swish = Activation(Activation.Swish, x)
      val expected = x * Activation(Activation.Sigmoid, x)
      assertEqualsDouble(swish, expected, tolerance, s"Swish mismatch at x=$x")
  }

  test("Swish(x) ≈ x for large positive x") {
    val result = Activation(Activation.Swish, 10.0)
    assertEqualsDouble(result, 10.0, 0.01)
  }

  test("Swish(x) ≈ 0 for large negative x") {
    val result = Activation(Activation.Swish, -10.0)
    assertEqualsDouble(result, 0.0, 0.01)
  }

  test("Swish derivative matches finite differences") {
    for x <- List(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0) do
      val analytical = Activation.deriv(Activation.Swish, x)
      val numerical = (Activation(Activation.Swish, x + fdEpsilon) - Activation(Activation.Swish, x - fdEpsilon)) / (2 * fdEpsilon)
      assertEqualsDouble(analytical, numerical, 1e-4,
        s"Swish deriv mismatch at x=$x: analytical=$analytical, numerical=$numerical")
  }

  // --- Ein integration: gradient through GELU/Swish layers ---

  test("GELU through Ein: gradient matches numerical") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(0.5, -0.3, 1.2)))
    val expr = Ein.Activate(Activation.GELU, v)
    verifyEinGrad(expr, "v", 3)
  }

  test("Swish through Ein: gradient matches numerical") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(0.5, -0.3, 1.2)))
    val expr = Ein.Activate(Activation.Swish, v)
    verifyEinGrad(expr, "v", 3)
  }

  test("dense layer with GELU: gradient matches numerical") {
    val features = Dim("j", 2)
    val out = Dim("o", 3)
    val w = Ein.Param("W", List(out, features),
      TensorData.fromArray(List(out, features), Array(0.1, 0.2, -0.3, 0.4, 0.5, -0.1)))
    val b = Ein.Param("b", List(out),
      TensorData.fromArray(List(out), Array(0.1, -0.1, 0.0)))
    val x = Ein.Param("x", List(features),
      TensorData.fromArray(List(features), Array(1.0, -0.5)))
    val expr = Ein.Activate(Activation.GELU, (w * x) + b)
    verifyEinGrad(expr, "W", 6)
    verifyEinGrad(expr, "b", 3)
  }

  // --- Helpers ---

  private val gradEpsilon = 1e-5
  private val gradTolerance = 1e-4

  private def perturbParam(
      expr: Ein[Double], paramId: String, elementIndex: Int, delta: Double
  ): Ein[Double] = expr match
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

  private def verifyEinGrad(
      expr: Ein[Double], paramId: String, paramSize: Int
  ): Unit =
    val loss = Ein.ReduceSum(expr, expr.outputDims.map(_.name))
    val grads = EinGrad.backward(loss, Map.empty)
    val grad = grads(paramId)
    for idx <- 0 until paramSize do
      val analytical = grad.data(idx)
      val plusExpr = perturbParam(loss, paramId, idx, gradEpsilon)
      val minusExpr = perturbParam(loss, paramId, idx, -gradEpsilon)
      val numerical = (EinEval.eval(plusExpr, Map.empty).data.sum - EinEval.eval(minusExpr, Map.empty).data.sum) / (2 * gradEpsilon)
      assertEqualsDouble(analytical, numerical, gradTolerance,
        s"Gradient mismatch for $paramId[$idx]: analytical=$analytical, numerical=$numerical")
