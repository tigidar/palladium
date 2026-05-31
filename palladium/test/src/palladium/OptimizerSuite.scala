package palladium

import palladium.Optimizer.SGD

class OptimizerSuite extends munit.FunSuite:

  val tolerance = 1e-9

  // ============================================
  // SGD basic parameter update tests
  // ============================================

  test("SGD: single parameter update") {
    val sgd = SGD[Double](learningRate = 0.1)
    val params = Map("x" -> 5.0)
    val grads = Map("x" -> 2.0)
    val updated = sgd.step(params, grads)
    // x_new = 5.0 - 0.1 * 2.0 = 4.8
    assertEqualsDouble(updated("x"), 4.8, tolerance)
  }

  test("SGD: multiple parameters update") {
    val sgd = SGD[Double](learningRate = 0.01)
    val params = Map("w1" -> 1.0, "w2" -> 2.0, "b" -> 0.5)
    val grads = Map("w1" -> 10.0, "w2" -> -5.0, "b" -> 2.0)
    val updated = sgd.step(params, grads)
    // w1_new = 1.0 - 0.01 * 10.0 = 0.9
    // w2_new = 2.0 - 0.01 * (-5.0) = 2.05
    // b_new = 0.5 - 0.01 * 2.0 = 0.48
    assertEqualsDouble(updated("w1"), 0.9, tolerance)
    assertEqualsDouble(updated("w2"), 2.05, tolerance)
    assertEqualsDouble(updated("b"), 0.48, tolerance)
  }

  test("SGD: zero gradient produces no change") {
    val sgd = SGD[Double](learningRate = 0.1)
    val params = Map("x" -> 3.0)
    val grads = Map("x" -> 0.0)
    val updated = sgd.step(params, grads)
    assertEqualsDouble(updated("x"), 3.0, tolerance)
  }

  test("SGD: missing gradient treated as zero") {
    val sgd = SGD[Double](learningRate = 0.1)
    val params = Map("x" -> 3.0, "y" -> 5.0)
    val grads = Map("x" -> 2.0) // no gradient for y
    val updated = sgd.step(params, grads)
    assertEqualsDouble(updated("x"), 2.8, tolerance)
    assertEqualsDouble(updated("y"), 5.0, tolerance) // unchanged
  }

  test("SGD: negative gradient increases parameter") {
    val sgd = SGD[Double](learningRate = 0.1)
    val params = Map("x" -> 1.0)
    val grads = Map("x" -> -3.0)
    val updated = sgd.step(params, grads)
    // x_new = 1.0 - 0.1 * (-3.0) = 1.3
    assertEqualsDouble(updated("x"), 1.3, tolerance)
  }

  // ============================================
  // SGD learning rate sensitivity tests
  // ============================================

  test("SGD: larger learning rate produces larger steps") {
    val params = Map("x" -> 10.0)
    val grads = Map("x" -> 5.0)

    val sgdSmall = SGD[Double](learningRate = 0.01)
    val sgdLarge = SGD[Double](learningRate = 0.1)

    val updatedSmall = sgdSmall.step(params, grads)
    val updatedLarge = sgdLarge.step(params, grads)

    // small: 10.0 - 0.01 * 5.0 = 9.95
    // large: 10.0 - 0.1 * 5.0 = 9.5
    assertEqualsDouble(updatedSmall("x"), 9.95, tolerance)
    assertEqualsDouble(updatedLarge("x"), 9.5, tolerance)
    assert(math.abs(10.0 - updatedLarge("x")) > math.abs(10.0 - updatedSmall("x")))
  }

  test("SGD: zero learning rate produces no change") {
    val sgd = SGD[Double](learningRate = 0.0)
    val params = Map("x" -> 5.0)
    val grads = Map("x" -> 100.0)
    val updated = sgd.step(params, grads)
    assertEqualsDouble(updated("x"), 5.0, tolerance)
  }

  // ============================================
  // SGD with Float type
  // ============================================

  test("SGD: works with Float type") {
    val sgd = SGD[Float](learningRate = 0.1f)
    val params = Map("x" -> 5.0f)
    val grads = Map("x" -> 2.0f)
    val updated = sgd.step(params, grads)
    assertEqualsFloat(updated("x"), 4.8f, tolerance.toFloat)
  }

  // ============================================
  // SGD gradient descent simulation
  // ============================================

  test("SGD: minimize f(x) = x^2 converges toward zero") {
    // Minimize x^2 starting from x=10
    // Gradient is 2x, so SGD step: x_new = x - lr * 2x = x(1 - 2*lr)
    var params = Map("x" -> 10.0)
    val sgd = SGD[Double](learningRate = 0.01)

    // Run 100 steps
    (0 until 100).foreach { _ =>
      val x = Value.variable("x", params("x"))
      val loss = x * x
      val grads = Grad.backward(loss)
      params = sgd.step(params, grads)
    }

    // After 100 steps with lr=0.01, should be much closer to 0
    // With lr=0.01 and grad=2x, multiplier is (1-0.02) = 0.98
    // After 100 steps: 10 * 0.98^100 ≈ 1.33
    assert(math.abs(params("x")) < 2.0, s"x should be near 0, got ${params("x")}")
    assert(math.abs(params("x")) < 10.0, "x should have decreased from initial value")
  }

  test("SGD: minimize f(x,y) = x^2 + y^2 converges both parameters") {
    var params = Map("x" -> 5.0, "y" -> -3.0)
    val sgd = SGD[Double](learningRate = 0.05)

    // Run 50 steps
    (0 until 50).foreach { _ =>
      val x = Value.variable("x", params("x"))
      val y = Value.variable("y", params("y"))
      val loss = x * x + y * y
      val grads = Grad.backward(loss)
      params = sgd.step(params, grads)
    }

    // Both should be closer to 0
    assert(math.abs(params("x")) < 5.0, s"x should have decreased")
    assert(math.abs(params("y")) < 3.0, s"y should have decreased")
  }

  // ============================================
  // SGD with MLP parameters format
  // ============================================

  test("SGD: works with MLP parameter naming convention") {
    val sgd = SGD[Double](learningRate = 0.01)
    val params = Map(
      "w0_0_0" -> 0.5,
      "w0_0_1" -> -0.3,
      "w0_1_0" -> 0.8,
      "w0_1_1" -> 0.1,
      "b0_0" -> 0.0,
      "b0_1" -> 0.0
    )
    val grads = Map(
      "w0_0_0" -> 1.0,
      "w0_0_1" -> -2.0,
      "w0_1_0" -> 0.5,
      "w0_1_1" -> -0.5,
      "b0_0" -> 0.1,
      "b0_1" -> -0.1
    )
    val updated = sgd.step(params, grads)

    assertEqualsDouble(updated("w0_0_0"), 0.5 - 0.01 * 1.0, tolerance)
    assertEqualsDouble(updated("w0_0_1"), -0.3 - 0.01 * (-2.0), tolerance)
    assertEqualsDouble(updated("w0_1_0"), 0.8 - 0.01 * 0.5, tolerance)
    assertEqualsDouble(updated("w0_1_1"), 0.1 - 0.01 * (-0.5), tolerance)
    assertEqualsDouble(updated("b0_0"), 0.0 - 0.01 * 0.1, tolerance)
    assertEqualsDouble(updated("b0_1"), 0.0 - 0.01 * (-0.1), tolerance)
  }

  test("SGD: integrates with MLP.withParams") {
    import palladium.nn.MLP

    val mlp = MLP.init[(2, 2, 1)](seed = 42L)
    val sgd = SGD[Double](learningRate = 0.01)

    val inputs = Vector(Value.Lit(1.0), Value.Lit(-0.5))
    val output = mlp.forward(inputs).head
    val grads = Grad.backward(output)

    val updatedParams = sgd.step(mlp.params, grads)
    val updatedMlp = mlp.withParams(updatedParams)

    // Verify parameters changed
    assertNotEquals(mlp.params, updatedMlp.params)

    // Verify all parameters are still present
    assertEquals(mlp.paramNames, updatedMlp.paramNames)
  }

  // ============================================
  // SGD syntax extension tests
  // ============================================

  test("SGD.syntax: apply method works") {
    import Optimizer.syntax.*

    val sgd = SGD[Double](learningRate = 0.1)
    val params = Map("x" -> 5.0)
    val grads = Map("x" -> 2.0)
    val updated = sgd(params, grads)
    assertEqualsDouble(updated("x"), 4.8, tolerance)
  }

  // ============================================
  // SGD multi-step update tests
  // ============================================

  test("SGD: multiple steps compound correctly") {
    val sgd = SGD[Double](learningRate = 0.1)
    var params = Map("x" -> 10.0)

    // Step 1: grad = 2
    params = sgd.step(params, Map("x" -> 2.0))
    assertEqualsDouble(params("x"), 9.8, tolerance)

    // Step 2: grad = 3
    params = sgd.step(params, Map("x" -> 3.0))
    assertEqualsDouble(params("x"), 9.5, tolerance)

    // Step 3: grad = -1
    params = sgd.step(params, Map("x" -> -1.0))
    assertEqualsDouble(params("x"), 9.6, tolerance)
  }

  test("SGD: preserves parameter keys across updates") {
    val sgd = SGD[Double](learningRate = 0.01)
    val params = Map("a" -> 1.0, "b" -> 2.0, "c" -> 3.0)
    val grads = Map("a" -> 0.5, "b" -> -0.5, "c" -> 0.0)
    val updated = sgd.step(params, grads)

    assertEquals(updated.keySet, params.keySet)
  }
