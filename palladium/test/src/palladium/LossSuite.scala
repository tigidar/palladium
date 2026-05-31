package palladium

class LossSuite extends munit.FunSuite:

  val epsilon = 1e-7
  val tolerance = 1e-5

  // ============================================
  // MSE Loss Tests
  // ============================================

  test("MSE: perfect predictions (zero loss)") {
    val predicted = Seq(
      Value.variable("p1", 1.0),
      Value.variable("p2", 2.0),
      Value.variable("p3", 3.0)
    )
    val target = Seq(
      Value.Lit(1.0),
      Value.Lit(2.0),
      Value.Lit(3.0)
    )

    val loss = Loss.mse(predicted, target)
    val value = loss.eval

    assertEqualsDouble(value, 0.0, tolerance)
  }

  test("MSE: simple difference") {
    // predicted = [2, 4], target = [1, 3]
    // errors = [1, 1], squared = [1, 1], mean = 1.0
    val predicted = Seq(
      Value.Lit(2.0),
      Value.Lit(4.0)
    )
    val target = Seq(
      Value.Lit(1.0),
      Value.Lit(3.0)
    )

    val loss = Loss.mse(predicted, target)
    val value = loss.eval

    assertEqualsDouble(value, 1.0, tolerance)
  }

  test("MSE: gradient check with variables") {
    // predicted = [x, y], target = [1, 2]
    // loss = (1/2) * [(x-1)² + (y-2)²]
    // d/dx = (1/2) * 2(x-1) = x-1 = 2-1 = 1
    // d/dy = (1/2) * 2(y-2) = y-2 = 4-2 = 2
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 4.0)
    val predicted = Seq(x, y)
    val target = Seq(Value.Lit(1.0), Value.Lit(2.0))

    val loss = Loss.mse(predicted, target)
    val grads = Grad.backward(loss)

    assertEqualsDouble(grads("x"), 1.0, tolerance)
    assertEqualsDouble(grads("y"), 2.0, tolerance)
  }

  test("MSE: gradient accumulation with shared variable") {
    // predicted = [x, x], target = [1, 2]
    // loss = (1/2) * [(x-1)² + (x-2)²]
    // d/dx = (1/2) * [2(x-1) + 2(x-2)]
    //      = (x-1) + (x-2) = 2x - 3
    // At x=3: 2*3 - 3 = 3
    val x = Value.variable("x", 3.0)
    val predicted = Seq(x, x)
    val target = Seq(Value.Lit(1.0), Value.Lit(2.0))

    val loss = Loss.mse(predicted, target)
    val grads = Grad.backward(loss)

    assertEqualsDouble(grads("x"), 3.0, tolerance)
  }

  test("MSE: numerical gradient verification") {
    val xVal = 2.5
    val yVal = 3.5
    // Manual MSE computation
    def f(x: Double, y: Double): Double =
      val e1 = (x - 1.0) * (x - 1.0)
      val e2 = (y - 2.0) * (y - 2.0)
      (e1 + e2) / 2.0

    val x = Value.variable("x", xVal)
    val y = Value.variable("y", yVal)
    val predicted = Seq(x, y)
    val target = Seq(Value.Lit(1.0), Value.Lit(2.0))

    val loss = Loss.mse(predicted, target)
    val grads = Grad.backward(loss)

    // Numerical gradients
    val h = 1e-7
    val numGradX = (f(xVal + h, yVal) - f(xVal - h, yVal)) / (2 * h)
    val numGradY = (f(xVal, yVal + h) - f(xVal, yVal - h)) / (2 * h)

    assertEqualsDouble(grads("x"), numGradX, tolerance)
    assertEqualsDouble(grads("y"), numGradY, tolerance)
  }

  test("MSE: single element") {
    val predicted = Seq(Value.variable("x", 5.0))
    val target = Seq(Value.Lit(3.0))

    val loss = Loss.mse(predicted, target)
    // loss = (5-3)² / 1 = 4
    assertEqualsDouble(loss.eval, 4.0, tolerance)

    val grads = Grad.backward(loss)
    // d/dx = 2(x-3)/1 = 2*2 = 4
    assertEqualsDouble(grads("x"), 4.0, tolerance)
  }

  test("MSE: requires equal lengths") {
    val predicted = Seq(Value.Lit(1.0), Value.Lit(2.0))
    val target = Seq(Value.Lit(1.0))

    interceptMessage[IllegalArgumentException](
      "requirement failed: predicted and target must have same length (got 2 vs 1)"
    ) {
      Loss.mse(predicted, target)
    }
  }

  test("MSE: requires non-empty sequences") {
    val predicted = Seq.empty[Value[Double]]
    val target = Seq.empty[Value[Double]]

    interceptMessage[IllegalArgumentException](
      "requirement failed: predicted and target cannot be empty"
    ) {
      Loss.mse(predicted, target)
    }
  }

  test("MSE: syntax extension method") {
    import Loss.syntax.*
    val predicted = Seq(Value.variable("x", 2.0))
    val target = Seq(Value.Lit(1.0))

    val loss1 = predicted.mse(target)
    val loss2 = Loss.mse(predicted, target)

    assertEqualsDouble(loss1.eval, loss2.eval, tolerance)
  }

  // ============================================
  // Binary Cross-Entropy Loss Tests
  // ============================================

  test("Binary cross-entropy: perfect predictions") {
    // BCE = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
    // When p=1, y=1: -[1*log(1) + 0*log(0)] = 0 (using limit)
    // When p=0, y=0: -[0*log(0) + 1*log(1)] = 0 (using limit)
    // In practice, we use values close to 0 and 1 to avoid log(0)
    val predicted = Seq(
      Value.Lit(0.99),
      Value.Lit(0.01)
    )
    val target = Seq(
      Value.Lit(1.0),
      Value.Lit(0.0)
    )

    val loss = Loss.binaryCrossEntropy(predicted, target)
    val value = loss.eval

    // Should be very close to 0
    assert(value < 0.02, s"Expected small loss, got $value")
  }

  test("Binary cross-entropy: worst case predictions") {
    // When p=0.01, y=1: -log(0.01) ≈ 4.605
    // When p=0.99, y=0: -log(0.01) ≈ 4.605
    val predicted = Seq(
      Value.Lit(0.01),
      Value.Lit(0.99)
    )
    val target = Seq(
      Value.Lit(1.0),
      Value.Lit(0.0)
    )

    val loss = Loss.binaryCrossEntropy(predicted, target)
    val value = loss.eval

    // Should be high loss (around 4.6)
    assert(value > 4.0, s"Expected high loss, got $value")
  }

  test("Binary cross-entropy: balanced predictions") {
    // When p=0.5: -[y*log(0.5) + (1-y)*log(0.5)] = -log(0.5) = log(2)
    val predicted = Seq(
      Value.Lit(0.5),
      Value.Lit(0.5)
    )
    val target = Seq(
      Value.Lit(1.0),
      Value.Lit(0.0)
    )

    val loss = Loss.binaryCrossEntropy(predicted, target)
    val value = loss.eval

    assertEqualsDouble(value, math.log(2.0), tolerance)
  }

  test("Binary cross-entropy: gradient check") {
    // BCE = -(1/2) * [y*log(p) + (1-y)*log(1-p)]
    // d/dp = -(1/2) * [y/p - (1-y)/(1-p)]
    val p = Value.variable("p", 0.6)
    val predicted = Seq(p)
    val target = Seq(Value.Lit(1.0))

    val loss = Loss.binaryCrossEntropy(predicted, target)
    val grads = Grad.backward(loss)

    // d/dp when y=1: -(1/1)*[1/p] = -1/0.6
    val expected = -1.0 / 0.6
    assertEqualsDouble(grads("p"), expected, tolerance)
  }

  test("Binary cross-entropy: gradient for negative class") {
    val p = Value.variable("p", 0.4)
    val predicted = Seq(p)
    val target = Seq(Value.Lit(0.0))

    val loss = Loss.binaryCrossEntropy(predicted, target)
    val grads = Grad.backward(loss)

    // d/dp when y=0: -(1/1)*[-(1-0)/(1-p)] = 1/(1-0.4) = 1/0.6
    val expected = 1.0 / (1.0 - 0.4)
    assertEqualsDouble(grads("p"), expected, tolerance)
  }

  test("Binary cross-entropy: numerical gradient verification") {
    val pVal = 0.7
    def f(p: Double): Double =
      val y = 1.0
      -(y * math.log(p) + (1 - y) * math.log(1 - p))

    val p = Value.variable("p", pVal)
    val predicted = Seq(p)
    val target = Seq(Value.Lit(1.0))

    val loss = Loss.binaryCrossEntropy(predicted, target)
    val grads = Grad.backward(loss)

    val h = 1e-7
    val numGrad = (f(pVal + h) - f(pVal - h)) / (2 * h)

    assertEqualsDouble(grads("p"), numGrad, tolerance)
  }

  test("Binary cross-entropy: multiple predictions gradient") {
    val p1 = Value.variable("p1", 0.8)
    val p2 = Value.variable("p2", 0.3)
    val predicted = Seq(p1, p2)
    val target = Seq(Value.Lit(1.0), Value.Lit(0.0))

    val loss = Loss.binaryCrossEntropy(predicted, target)
    val grads = Grad.backward(loss)

    // d/dp1 when y1=1: -(1/2)*[1/p1] = -1/(2*0.8)
    val expected1 = -1.0 / (2.0 * 0.8)
    // d/dp2 when y2=0: -(1/2)*[-1/(1-p2)] = 1/(2*(1-0.3))
    val expected2 = 1.0 / (2.0 * (1.0 - 0.3))

    assertEqualsDouble(grads("p1"), expected1, tolerance)
    assertEqualsDouble(grads("p2"), expected2, tolerance)
  }

  test("Binary cross-entropy: requires equal lengths") {
    val predicted = Seq(Value.Lit(0.5), Value.Lit(0.5))
    val target = Seq(Value.Lit(1.0))

    interceptMessage[IllegalArgumentException](
      "requirement failed: predicted and target must have same length (got 2 vs 1)"
    ) {
      Loss.binaryCrossEntropy(predicted, target)
    }
  }

  test("Binary cross-entropy: syntax extension method") {
    import Loss.syntax.*
    val predicted = Seq(Value.Lit(0.7))
    val target = Seq(Value.Lit(1.0))

    val loss1 = predicted.binaryCrossEntropy(target)
    val loss2 = Loss.binaryCrossEntropy(predicted, target)

    assertEqualsDouble(loss1.eval, loss2.eval, tolerance)
  }

  // ============================================
  // Categorical Cross-Entropy Loss Tests
  // ============================================

  test("Categorical cross-entropy: perfect predictions") {
    // One sample, 3 classes, one-hot target [1, 0, 0], predicted [0.99, 0.005, 0.005]
    val predicted = Seq(
      Seq(Value.Lit(0.99), Value.Lit(0.005), Value.Lit(0.005))
    )
    val target = Seq(
      Seq(Value.Lit(1.0), Value.Lit(0.0), Value.Lit(0.0))
    )

    val loss = Loss.categoricalCrossEntropy(predicted, target)
    val value = loss.eval

    // loss = -log(0.99) ≈ 0.01
    assert(value < 0.02, s"Expected small loss, got $value")
  }

  test("Categorical cross-entropy: worst case predictions") {
    // Target is class 0, but model predicts class 2
    val predicted = Seq(
      Seq(Value.Lit(0.01), Value.Lit(0.01), Value.Lit(0.98))
    )
    val target = Seq(
      Seq(Value.Lit(1.0), Value.Lit(0.0), Value.Lit(0.0))
    )

    val loss = Loss.categoricalCrossEntropy(predicted, target)
    val value = loss.eval

    // loss = -log(0.01) ≈ 4.605
    assert(value > 4.0, s"Expected high loss, got $value")
  }

  test("Categorical cross-entropy: uniform predictions") {
    // 3 classes, uniform probability 1/3 each
    val predicted = Seq(
      Seq(Value.Lit(1.0 / 3), Value.Lit(1.0 / 3), Value.Lit(1.0 / 3))
    )
    val target = Seq(
      Seq(Value.Lit(1.0), Value.Lit(0.0), Value.Lit(0.0))
    )

    val loss = Loss.categoricalCrossEntropy(predicted, target)
    val value = loss.eval

    // loss = -log(1/3) = log(3)
    assertEqualsDouble(value, math.log(3.0), tolerance)
  }

  test("Categorical cross-entropy: multiple samples") {
    // 2 samples, 2 classes each
    // Sample 1: target class 0, predicted [0.7, 0.3]
    // Sample 2: target class 1, predicted [0.4, 0.6]
    val predicted = Seq(
      Seq(Value.Lit(0.7), Value.Lit(0.3)),
      Seq(Value.Lit(0.4), Value.Lit(0.6))
    )
    val target = Seq(
      Seq(Value.Lit(1.0), Value.Lit(0.0)),
      Seq(Value.Lit(0.0), Value.Lit(1.0))
    )

    val loss = Loss.categoricalCrossEntropy(predicted, target)
    val value = loss.eval

    // loss = -(1/2) * [log(0.7) + log(0.6)]
    val expected = -(math.log(0.7) + math.log(0.6)) / 2.0
    assertEqualsDouble(value, expected, tolerance)
  }

  test("Categorical cross-entropy: gradient check") {
    // Single sample, 2 classes, target [1, 0]
    val p1 = Value.variable("p1", 0.7)
    val p2 = Value.variable("p2", 0.3)
    val predicted = Seq(Seq(p1, p2))
    val target = Seq(Seq(Value.Lit(1.0), Value.Lit(0.0)))

    val loss = Loss.categoricalCrossEntropy(predicted, target)
    val grads = Grad.backward(loss)

    // loss = -log(p1), since target is [1, 0]
    // d/dp1 = -(1/1) * (1/p1) = -1/0.7
    // d/dp2 = -(1/1) * (0/p2) = 0
    assertEqualsDouble(grads("p1"), -1.0 / 0.7, tolerance)
    assertEqualsDouble(grads("p2"), 0.0, tolerance)
  }

  test("Categorical cross-entropy: gradient with multiple classes") {
    val p1 = Value.variable("p1", 0.5)
    val p2 = Value.variable("p2", 0.3)
    val p3 = Value.variable("p3", 0.2)
    val predicted = Seq(Seq(p1, p2, p3))
    val target = Seq(Seq(Value.Lit(0.0), Value.Lit(1.0), Value.Lit(0.0)))

    val loss = Loss.categoricalCrossEntropy(predicted, target)
    val grads = Grad.backward(loss)

    // loss = -log(p2), since target is [0, 1, 0]
    // d/dp1 = 0, d/dp2 = -1/p2, d/dp3 = 0
    assertEqualsDouble(grads("p1"), 0.0, tolerance)
    assertEqualsDouble(grads("p2"), -1.0 / 0.3, tolerance)
    assertEqualsDouble(grads("p3"), 0.0, tolerance)
  }

  test("Categorical cross-entropy: numerical gradient verification") {
    val p1Val = 0.6
    val p2Val = 0.4
    def f(p1: Double, p2: Double): Double =
      // target = [1, 0], so loss = -log(p1)
      -math.log(p1)

    val p1 = Value.variable("p1", p1Val)
    val p2 = Value.variable("p2", p2Val)
    val predicted = Seq(Seq(p1, p2))
    val target = Seq(Seq(Value.Lit(1.0), Value.Lit(0.0)))

    val loss = Loss.categoricalCrossEntropy(predicted, target)
    val grads = Grad.backward(loss)

    val h = 1e-7
    val numGradP1 = (f(p1Val + h, p2Val) - f(p1Val - h, p2Val)) / (2 * h)

    assertEqualsDouble(grads("p1"), numGradP1, tolerance)
  }

  test("Categorical cross-entropy: requires equal sample counts") {
    val predicted = Seq(
      Seq(Value.Lit(0.5), Value.Lit(0.5))
    )
    val target = Seq(
      Seq(Value.Lit(1.0), Value.Lit(0.0)),
      Seq(Value.Lit(0.0), Value.Lit(1.0))
    )

    interceptMessage[IllegalArgumentException](
      "requirement failed: predicted and target must have same length (got 1 vs 2)"
    ) {
      Loss.categoricalCrossEntropy(predicted, target)
    }
  }

  test("Categorical cross-entropy: requires equal class counts") {
    val predicted = Seq(
      Seq(Value.Lit(0.5), Value.Lit(0.5))
    )
    val target = Seq(
      Seq(Value.Lit(1.0), Value.Lit(0.0), Value.Lit(0.0))
    )

    interceptMessage[IllegalArgumentException](
      "requirement failed: predicted and target must have same number of classes (got 2 vs 3)"
    ) {
      Loss.categoricalCrossEntropy(predicted, target)
    }
  }

  test("Categorical cross-entropy: syntax extension method") {
    import Loss.syntax.*
    val predicted = Seq(Seq(Value.Lit(0.7), Value.Lit(0.3)))
    val target = Seq(Seq(Value.Lit(1.0), Value.Lit(0.0)))

    val loss1 = predicted.categoricalCrossEntropy(target)
    val loss2 = Loss.categoricalCrossEntropy(predicted, target)

    assertEqualsDouble(loss1.eval, loss2.eval, tolerance)
  }

  // ============================================
  // Integration Tests
  // ============================================

  test("MSE with sigmoid predictions (regression)") {
    // Simulating a simple neural network output
    val w = Value.variable("w", 0.5)
    val b = Value.variable("b", 0.1)
    val x = Value.Lit(2.0)

    // predicted = sigmoid(w*x + b)
    val predicted = Seq((w * x + b).sigmoid)
    val target = Seq(Value.Lit(0.8))

    val loss = Loss.mse(predicted, target)
    val value = loss.eval
    val grads = Grad.backward(loss)

    // Just verify that gradients exist and are finite
    assert(grads.contains("w"))
    assert(grads.contains("b"))
    assert(!grads("w").isNaN && !grads("w").isInfinite)
    assert(!grads("b").isNaN && !grads("b").isInfinite)
  }

  test("Binary cross-entropy with sigmoid predictions") {
    // Typical usage: logits -> sigmoid -> BCE
    val w = Value.variable("w", 0.5)
    val x = Value.Lit(2.0)

    // logit = w * x
    // predicted probability = sigmoid(logit)
    val logit = w * x
    val predicted = Seq(logit.sigmoid)
    val target = Seq(Value.Lit(1.0))

    val loss = Loss.binaryCrossEntropy(predicted, target)
    val grads = Grad.backward(loss)

    // Gradient should exist and be finite
    assert(grads.contains("w"))
    assert(!grads("w").isNaN && !grads("w").isInfinite)
  }

  test("AutoGrad integration with MSE") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val predicted = Seq(x, y)
    val target = Seq(Value.Lit(1.0), Value.Lit(2.0))

    val loss = Loss.mse(predicted, target)
    val result = AutoGrad.trace(loss)

    // Verify both value and gradients are computed
    assert(result.value > 0)
    assertEquals(result.grads("x"), 1.0)
    assertEquals(result.grads("y"), 1.0)
  }
