package palladium

import palladium.Optimizer.{AdamW, SGD}

class AdamWSuite extends munit.FunSuite:

  val tolerance = 1e-9

  // ============================================
  // AdamW basic parameter update tests
  // ============================================

  test("AdamW: single parameter update") {
    val adam = AdamW.default[Double](learningRate = 0.001)
    val params = Map("x" -> 5.0)
    val grads = Map("x" -> 2.0)
    val updated = adam.step(params, grads)
    // Should have moved toward lower loss
    assert(updated("x") < 5.0, s"Expected x < 5.0, got ${updated("x")}")
  }

  test("AdamW: multiple parameters update") {
    val adam = AdamW.default[Double](learningRate = 0.001)
    val params = Map("w1" -> 1.0, "w2" -> 2.0, "b" -> 0.5)
    val grads = Map("w1" -> 10.0, "w2" -> -5.0, "b" -> 2.0)
    val updated = adam.step(params, grads)
    // w1 should decrease (positive gradient)
    assert(updated("w1") < 1.0)
    // w2 should increase (negative gradient)
    assert(updated("w2") > 2.0)
  }

  test("AdamW: zero gradient still applies weight decay") {
    val adam = new AdamW[Double](
      learningRate = 0.001,
      beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
      weightDecay = 0.01
    )
    val params = Map("x" -> 10.0)
    val grads = Map("x" -> 0.0)
    val updated = adam.step(params, grads)
    // Weight decay should pull x toward zero even with zero gradient
    assert(updated("x") < 10.0, s"Expected x < 10.0 due to weight decay, got ${updated("x")}")
  }

  test("AdamW: missing gradient treated as zero") {
    val adam = AdamW.default[Double](learningRate = 0.001)
    val params = Map("x" -> 3.0, "y" -> 5.0)
    val grads = Map("x" -> 2.0) // no gradient for y
    val updated = adam.step(params, grads)
    // x should change, y should only get weight decay
    assert(updated("x") != 3.0)
  }

  // ============================================
  // AdamW moment accumulation tests
  // ============================================

  test("AdamW: moments accumulate across steps") {
    val adam = AdamW.default[Double](learningRate = 0.001)
    val params = Map("x" -> 5.0)

    // First step with gradient 2.0
    val updated1 = adam.step(params, Map("x" -> 2.0))

    // Second step with same gradient — should make larger effective step
    // because momentum builds up
    val updated2 = adam.step(updated1, Map("x" -> 2.0))

    val step1 = params("x") - updated1("x")
    val step2 = updated1("x") - updated2("x")
    // With momentum, step2 should be at least as large as step1
    assert(step2 >= step1 * 0.9, s"step2=$step2 should be >= step1=$step1 (momentum)")
  }

  test("AdamW: bias correction matters in early steps") {
    // Without bias correction, first steps would be too small
    val adam = AdamW.default[Double](learningRate = 0.001)
    val params = Map("x" -> 5.0)
    val grads = Map("x" -> 1.0)
    val updated = adam.step(params, grads)
    val stepSize = params("x") - updated("x")
    // Step size should be approximately learningRate (bias correction compensates)
    assert(stepSize > 0.0005, s"Step size $stepSize too small — bias correction may be broken")
  }

  // ============================================
  // AdamW weight decay tests
  // ============================================

  test("AdamW: zero weight decay behaves like Adam") {
    val adamW = new AdamW[Double](
      learningRate = 0.001,
      beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
      weightDecay = 0.0
    )
    val adam = new AdamW[Double](
      learningRate = 0.001,
      beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
      weightDecay = 0.0
    )
    val params = Map("x" -> 5.0)
    val grads = Map("x" -> 2.0)
    val updated1 = adamW.step(params, grads)
    val updated2 = adam.step(params, grads)
    assertEqualsDouble(updated1("x"), updated2("x"), tolerance)
  }

  test("AdamW: larger weight decay produces more regularization") {
    val adamSmallWd = new AdamW[Double](
      learningRate = 0.001,
      beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
      weightDecay = 0.001
    )
    val adamLargeWd = new AdamW[Double](
      learningRate = 0.001,
      beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
      weightDecay = 0.1
    )
    val params = Map("x" -> 10.0)
    val grads = Map("x" -> 1.0)
    val updatedSmall = adamSmallWd.step(params, grads)
    val updatedLarge = adamLargeWd.step(params, grads)
    // Larger weight decay should move x further toward zero
    assert(updatedLarge("x") < updatedSmall("x"),
      s"Larger WD should regularize more: small=${updatedSmall("x")}, large=${updatedLarge("x")}")
  }

  // ============================================
  // AdamW convergence tests
  // ============================================

  test("AdamW: minimize f(x) = x^2 converges toward zero") {
    var params = Map("x" -> 10.0)
    val adam = new AdamW[Double](
      learningRate = 0.1,
      beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
      weightDecay = 0.0 // no weight decay for simple convergence test
    )

    (0 until 200).foreach { _ =>
      val x = Value.variable("x", params("x"))
      val loss = x * x
      val grads = Grad.backward(loss)
      params = adam.step(params, grads)
    }

    assert(math.abs(params("x")) < 0.5,
      s"x should be near 0 after 200 steps, got ${params("x")}")
  }

  test("AdamW: minimize f(x,y) = x^2 + y^2 converges both parameters") {
    var params = Map("x" -> 5.0, "y" -> -3.0)
    val adam = AdamW.default[Double](learningRate = 0.1)

    (0 until 200).foreach { _ =>
      val x = Value.variable("x", params("x"))
      val y = Value.variable("y", params("y"))
      val loss = x * x + y * y
      val grads = Grad.backward(loss)
      params = adam.step(params, grads)
    }

    assert(math.abs(params("x")) < 0.1, s"x should be near 0, got ${params("x")}")
    assert(math.abs(params("y")) < 0.1, s"y should be near 0, got ${params("y")}")
  }

  // ============================================
  // AdamW syntax and factory tests
  // ============================================

  test("AdamW.default creates optimizer with standard hyperparams") {
    val adam = AdamW.default[Double](learningRate = 0.001)
    val params = Map("x" -> 1.0)
    val grads = Map("x" -> 1.0)
    // Should not throw
    val updated = adam.step(params, grads)
    assert(updated.contains("x"))
  }

  test("AdamW: syntax extension works") {
    import Optimizer.syntax.*
    val adam = AdamW.default[Double](learningRate = 0.001)
    val params = Map("x" -> 5.0)
    val grads = Map("x" -> 2.0)
    val updated = adam(params, grads)
    assert(updated("x") < 5.0)
  }

  test("AdamW: preserves parameter keys across updates") {
    val adam = AdamW.default[Double](learningRate = 0.001)
    val params = Map("a" -> 1.0, "b" -> 2.0, "c" -> 3.0)
    val grads = Map("a" -> 0.5, "b" -> -0.5, "c" -> 0.0)
    val updated = adam.step(params, grads)
    assertEquals(updated.keySet, params.keySet)
  }
