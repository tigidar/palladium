package palladium.nn

import palladium.{Grad, NumberLike, Value}

class MLPSuite extends munit.FunSuite:

  val tolerance = 1e-5

  // ============================================
  // Parameter count verification
  // ============================================

  test("parameter count: MLP[(2, 3, 1)] = (2*3+3) + (3*1+1) = 13") {
    val mlp = MLP.init[(2, 3, 1)](seed = 42L)
    assertEquals(mlp.paramCount, 13)
  }

  test("parameter count: MLP[(3, 4, 4, 1)] = 16 + 20 + 5 = 41") {
    val mlp = MLP.init[(3, 4, 4, 1)](seed = 42L)
    assertEquals(mlp.paramCount, 41)
  }

  test("parameter count: MLP[(1, 1)] = 1*1+1 = 2") {
    val mlp = MLP.init[(1, 1)](seed = 42L)
    assertEquals(mlp.paramCount, 2)
  }

  // ============================================
  // Deterministic initialization
  // ============================================

  test("deterministic init: same seed produces identical params") {
    val mlp1 = MLP.init[(3, 4, 1)](seed = 123L)
    val mlp2 = MLP.init[(3, 4, 1)](seed = 123L)
    assertEquals(mlp1.params, mlp2.params)
  }

  test("deterministic init: different seeds produce different params") {
    val mlp1 = MLP.init[(3, 4, 1)](seed = 1L)
    val mlp2 = MLP.init[(3, 4, 1)](seed = 2L)
    assertNotEquals(mlp1.params, mlp2.params)
  }

  // ============================================
  // Forward pass output size
  // ============================================

  test("forward output size matches last layer dimension") {
    val mlp = MLP.init[(3, 4, 4, 1)](seed = 42L)
    val inputs = Vector(Value.Lit(0.5), Value.Lit(0.3), Value.Lit(-0.2))
    val outputs = mlp.forward(inputs)
    assertEquals(outputs.size, 1)
  }

  test("forward output size: multi-output MLP[(2, 3, 2)]") {
    val mlp = MLP.init[(2, 3, 2)](seed = 42L)
    val inputs = Vector(Value.Lit(1.0), Value.Lit(2.0))
    val outputs = mlp.forward(inputs)
    assertEquals(outputs.size, 2)
  }

  // ============================================
  // Forward pass produces finite values
  // ============================================

  test("forward pass produces finite eval values") {
    val mlp = MLP.init[(3, 4, 4, 1)](seed = 42L)
    val inputs = Vector(Value.Lit(0.5), Value.Lit(0.3), Value.Lit(-0.2))
    val outputs = mlp.forward(inputs)
    outputs.foreach { out =>
      val v = out.eval
      assert(!v.isNaN, s"output was NaN")
      assert(!v.isInfinite, s"output was Infinite")
    }
  }

  // ============================================
  // Input size validation
  // ============================================

  test("forward rejects wrong input size") {
    val mlp = MLP.init[(3, 4, 1)](seed = 42L)
    val wrongInputs = Vector(Value.Lit(1.0), Value.Lit(2.0)) // 2 instead of 3
    interceptMessage[IllegalArgumentException]("requirement failed: Expected 3 inputs, got 2") {
      mlp.forward(wrongInputs)
    }
  }

  // ============================================
  // Gradients computable for all params
  // ============================================

  test("gradients computable via Grad.backward for all params") {
    val mlp = MLP.init[(2, 3, 1)](seed = 42L)
    val inputs = Vector(Value.Lit(0.5), Value.Lit(-0.3))
    val output = mlp.forward(inputs).head
    val grads = Grad.backward(output)
    mlp.paramNames.foreach { name =>
      assert(grads.contains(name), s"missing gradient for $name")
    }
  }

  // ============================================
  // All gradients are finite
  // ============================================

  test("all gradients are finite") {
    val mlp = MLP.init[(3, 4, 4, 1)](seed = 42L)
    val inputs = Vector(Value.Lit(0.5), Value.Lit(0.3), Value.Lit(-0.2))
    val output = mlp.forward(inputs).head
    val grads = Grad.backward(output)
    grads.foreach { case (name, grad) =>
      assert(!grad.isNaN, s"gradient for $name was NaN")
      assert(!grad.isInfinite, s"gradient for $name was Infinite")
    }
  }

  // ============================================
  // Numerical gradient verification (finite differences)
  // ============================================

  test("numerical gradient verification for MLP[(2, 3, 1)]") {
    val mlp = MLP.init[(2, 3, 1)](seed = 42L)
    val inputValues = Vector(0.5, -0.3)
    val inputs = inputValues.map(Value.Lit(_))
    val output = mlp.forward(inputs).head
    val analyticGrads = Grad.backward(output)

    val h = 1e-5
    mlp.paramNames.foreach { paramName =>
      val originalValue = mlp.params(paramName)

      val plusMlp = mlp.withParams(Map(paramName -> (originalValue + h)))
      val plusOut = plusMlp.forward(inputValues.map(Value.Lit(_))).head.eval

      val minusMlp = mlp.withParams(Map(paramName -> (originalValue - h)))
      val minusOut = minusMlp.forward(inputValues.map(Value.Lit(_))).head.eval

      val numericalGrad = (plusOut - minusOut) / (2 * h)
      val analyticGrad = analyticGrads(paramName)

      assertEqualsDouble(
        analyticGrad,
        numericalGrad,
        1e-4,
        s"gradient mismatch for $paramName: analytic=$analyticGrad, numerical=$numericalGrad"
      )
    }
  }

  // ============================================
  // withParams changes forward pass results
  // ============================================

  test("withParams changes forward pass output") {
    val mlp = MLP.init[(2, 3, 1)](seed = 42L)
    val inputs = Vector(Value.Lit(1.0), Value.Lit(1.0))
    val before = mlp.forward(inputs).head.eval

    val tweaked = mlp.withParams(Map("w0_0_0" -> 999.0))
    val after = tweaked.forward(inputs).head.eval

    assertNotEquals(before, after)
  }

  // ============================================
  // Xavier init bounds check
  // ============================================

  test("Xavier init: all weights within expected bounds") {
    val mlp = MLP.init[(3, 4, 4, 1)](seed = 42L)
    val layers = Vector(3, 4, 4, 1)
    val layerPairs = layers.sliding(2).toVector.zipWithIndex

    layerPairs.foreach { case (pair, layerIdx) =>
      val fanIn = pair(0)
      val fanOut = pair(1)
      val limit = math.sqrt(6.0 / (fanIn + fanOut))

      for
        i <- 0 until fanIn
        j <- 0 until fanOut
      do
        val w = mlp.params(s"w${layerIdx}_${i}_$j")
        assert(
          w >= -limit && w <= limit,
          s"w${layerIdx}_${i}_$j = $w outside [-$limit, $limit]"
        )
    }
  }

  test("Xavier init: all biases are zero") {
    val mlp = MLP.init[(3, 4, 4, 1)](seed = 42L)
    mlp.paramNames.filter(_.startsWith("b")).foreach { name =>
      assertEquals(mlp.params(name), 0.0, s"bias $name should be 0.0")
    }
  }
