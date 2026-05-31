package palladium.ein

class EinLossSuite extends munit.FunSuite:

  val tolerance = 1e-5
  val epsilon = 1e-5

  // --- Cross-entropy forward tests ---

  test("cross-entropy: perfect prediction gives near-zero loss") {
    val classes = Dim("c", 3)
    // Logits strongly favoring class 0
    val logits = Ein.Param("logits", List(classes),
      TensorData.fromArray(List(classes), Array(10.0, -10.0, -10.0)))
    // One-hot target for class 0
    val targets = Ein.Param("targets", List(classes),
      TensorData.fromArray(List(classes), Array(1.0, 0.0, 0.0)))

    val loss = EinLoss.crossEntropy(logits, targets, "c")
    val result = EinEval.eval(loss, Map.empty)
    // Should be near 0 since softmax(logits) ≈ [1, 0, 0]
    assert(result.data(0) < 0.01, s"Expected near-zero loss, got ${result.data(0)}")
    assert(result.data(0) >= 0.0, s"Loss should be non-negative, got ${result.data(0)}")
  }

  test("cross-entropy: uniform logits with one-hot target gives log(numClasses)") {
    val classes = Dim("c", 3)
    // Uniform logits → softmax = [1/3, 1/3, 1/3] → log_softmax = [-log(3), ...]
    val logits = Ein.Param("logits", List(classes),
      TensorData.fromArray(List(classes), Array(0.0, 0.0, 0.0)))
    val targets = Ein.Param("targets", List(classes),
      TensorData.fromArray(List(classes), Array(1.0, 0.0, 0.0)))

    val loss = EinLoss.crossEntropy(logits, targets, "c")
    val result = EinEval.eval(loss, Map.empty)
    // CE = -1 * (-log(3)) = log(3) ≈ 1.0986
    assertEqualsDouble(result.data(0), math.log(3.0), 1e-6)
  }

  test("cross-entropy: worst prediction gives high loss") {
    val classes = Dim("c", 3)
    // Logits strongly favoring class 2, but target is class 0
    val logits = Ein.Param("logits", List(classes),
      TensorData.fromArray(List(classes), Array(-10.0, -10.0, 10.0)))
    val targets = Ein.Param("targets", List(classes),
      TensorData.fromArray(List(classes), Array(1.0, 0.0, 0.0)))

    val loss = EinLoss.crossEntropy(logits, targets, "c")
    val result = EinEval.eval(loss, Map.empty)
    // Loss should be large (≈ 20)
    assert(result.data(0) > 10.0, s"Expected high loss, got ${result.data(0)}")
  }

  test("cross-entropy: 2D batch of logits") {
    val batch = Dim("b", 2)
    val classes = Dim("c", 3)
    val logits = Ein.Param("logits", List(batch, classes),
      TensorData.fromArray(List(batch, classes), Array(
        2.0, 1.0, 0.0,   // sample 0: favors class 0
        0.0, 0.0, 2.0    // sample 1: favors class 2
      )))
    val targets = Ein.Param("targets", List(batch, classes),
      TensorData.fromArray(List(batch, classes), Array(
        1.0, 0.0, 0.0,   // target: class 0
        0.0, 0.0, 1.0    // target: class 2
      )))

    val loss = EinLoss.crossEntropy(logits, targets, "c")
    val result = EinEval.eval(loss, Map.empty)
    // Both predictions are reasonable, so total loss should be moderate
    assert(result.data(0) > 0.0, "Loss should be positive")
    assert(result.data(0) < 5.0, s"Loss should be moderate, got ${result.data(0)}")
  }

  // --- Cross-entropy gradient tests ---

  test("cross-entropy gradient matches numerical (1D)") {
    val classes = Dim("c", 3)
    val logits = Ein.Param("logits", List(classes),
      TensorData.fromArray(List(classes), Array(1.0, 0.5, -0.5)))
    val targets = Ein.Param("targets", List(classes),
      TensorData.fromArray(List(classes), Array(1.0, 0.0, 0.0)))

    val loss = EinLoss.crossEntropy(logits, targets, "c")
    verifyGrad(loss, "logits", 3)
  }

  test("cross-entropy gradient matches numerical (2D batch)") {
    val batch = Dim("b", 2)
    val classes = Dim("c", 3)
    val logits = Ein.Param("logits", List(batch, classes),
      TensorData.fromArray(List(batch, classes), Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    val targets = Ein.Param("targets", List(batch, classes),
      TensorData.fromArray(List(batch, classes), Array(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)))

    val loss = EinLoss.crossEntropy(logits, targets, "c")
    verifyGrad(loss, "logits", 6)
  }

  test("cross-entropy gradient: softmax(logits) - targets (classic result)") {
    // For CE loss with one-hot targets, d(loss)/d(logits) = softmax(logits) - targets
    val classes = Dim("c", 3)
    val logits = Ein.Param("logits", List(classes),
      TensorData.fromArray(List(classes), Array(1.0, 2.0, 3.0)))
    val targets = Ein.Param("targets", List(classes),
      TensorData.fromArray(List(classes), Array(0.0, 1.0, 0.0)))

    val loss = EinLoss.crossEntropy(logits, targets, "c")
    val grads = EinGrad.backward(loss, Map.empty)
    val logitsGrad = grads("logits")

    // Expected: softmax([1,2,3]) - [0,1,0]
    val softmaxResult = EinEval.eval(Ein.Softmax(logits, "c"), Map.empty)
    for i <- 0 until 3 do
      val expected = softmaxResult.data(i) - targets.asInstanceOf[Ein.Param[Double]].data.data(i)
      assertEqualsDouble(logitsGrad.data(i), expected, 1e-6,
        s"Gradient[$i] should be softmax - target: got ${logitsGrad.data(i)}, expected $expected")
  }

  // --- MSE forward tests ---

  test("mse: zero error gives zero loss") {
    val d = Dim("d", 3)
    val pred = Ein.Param("pred", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val target = Ein.Param("target", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))

    val loss = EinLoss.mse(pred, target)
    val result = EinEval.eval(loss, Map.empty)
    assertEqualsDouble(result.data(0), 0.0, 1e-10)
  }

  test("mse: known difference") {
    val d = Dim("d", 2)
    val pred = Ein.Param("pred", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0)))
    val target = Ein.Param("target", List(d), TensorData.fromArray(List(d), Array(3.0, 5.0)))

    val loss = EinLoss.mse(pred, target)
    val result = EinEval.eval(loss, Map.empty)
    // (1-3)^2 + (2-5)^2 = 4 + 9 = 13
    assertEqualsDouble(result.data(0), 13.0, 1e-10)
  }

  test("mse gradient matches numerical") {
    val d = Dim("d", 3)
    val pred = Ein.Param("pred", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val target = Ein.Param("target", List(d), TensorData.fromArray(List(d), Array(1.5, 2.5, 2.0)))

    val loss = EinLoss.mse(pred, target)
    verifyGrad(loss, "pred", 3)
  }

  // --- DSL integration ---

  test("crossEntropyLoss extension works") {
    import EinDsl.*
    val classes = Dim("c", 3)
    val logits = Ein.Param[Double]("logits", List(classes),
      TensorData.fromArray(List(classes), Array(1.0, 2.0, 3.0)))
    val targets = Ein.Param[Double]("targets", List(classes),
      TensorData.fromArray(List(classes), Array(0.0, 1.0, 0.0)))

    val loss1 = logits.crossEntropyLoss(targets, "c")
    val loss2 = EinLoss.crossEntropy(logits, targets, "c")
    val r1 = EinEval.eval(loss1, Map.empty)
    val r2 = EinEval.eval(loss2, Map.empty)
    assertEqualsDouble(r1.data(0), r2.data(0), 1e-10)
  }

  // --- Helpers ---

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

  private def verifyGrad(
      lossExpr: Ein[Double], paramId: String, paramSize: Int
  ): Unit =
    // Loss is already scalar — use it directly
    val grads = EinGrad.backward(lossExpr, Map.empty)
    val grad = grads(paramId)
    for idx <- 0 until paramSize do
      val analytical = grad.data(idx)
      val plusExpr = perturbParam(lossExpr, paramId, idx, epsilon)
      val minusExpr = perturbParam(lossExpr, paramId, idx, -epsilon)
      val numerical = (EinEval.eval(plusExpr, Map.empty).data.sum - EinEval.eval(minusExpr, Map.empty).data.sum) / (2 * epsilon)
      assertEqualsDouble(analytical, numerical, tolerance,
        s"Gradient mismatch for $paramId[$idx]: analytical=$analytical, numerical=$numerical")
