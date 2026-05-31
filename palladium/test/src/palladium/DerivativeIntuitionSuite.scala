package palladium

class DerivativeIntuitionSuite extends munit.FunSuite:

  // Polynomial: f(x) = 3x^2 - 4x + 5
  // True derivative: f'(x) = 6x - 4
  def f(x: Value[Double]): Value[Double] =
    Value.const(3) * (x ~^ Value.const(2)) - Value.const(4) * x + Value.const(5)

  // Numerical derivative: (f(x + h) - f(x)) / h
  def numericalDerivative(x: Double, h: Double): Double =
    (f(Value(x + h)).eval - f(Value(x)).eval) / h

  test("numerical derivative at x=2 is positive (converges to 8)") {
    // f'(2) = 6(2) - 4 = 8
    val h1 = numericalDerivative(2.0, 0.1)
    val h2 = numericalDerivative(2.0, 0.01)
    val h3 = numericalDerivative(2.0, 0.001)
    val h4 = numericalDerivative(2.0, 0.0001)

    // As h shrinks, we converge to 8
    assertEqualsDouble(h1, 8.3, 0.01)
    assertEqualsDouble(h2, 8.03, 0.001)
    assertEqualsDouble(h3, 8.003, 0.0001)
    assertEqualsDouble(h4, 8.0003, 0.00001)

    // All positive
    assert(h4 > 0, "Derivative at x=2 should be positive")
  }

  test("numerical derivative at x=0 is negative (converges to -4)") {
    // f'(0) = 6(0) - 4 = -4
    val h1 = numericalDerivative(0.0, 0.1)
    val h2 = numericalDerivative(0.0, 0.01)
    val h3 = numericalDerivative(0.0, 0.001)
    val h4 = numericalDerivative(0.0, 0.0001)

    // As h shrinks, we converge to -4
    assertEqualsDouble(h1, -3.7, 0.01)
    assertEqualsDouble(h2, -3.97, 0.001)
    assertEqualsDouble(h3, -3.997, 0.0001)
    assertEqualsDouble(h4, -3.9997, 0.00001)

    // All negative
    assert(h4 < 0, "Derivative at x=0 should be negative")
  }

  test("numerical derivative at x=2/3 is near zero (minimum of parabola)") {
    // f'(2/3) = 6(2/3) - 4 = 4 - 4 = 0
    // This is where the parabola has its minimum
    val x = 2.0 / 3.0
    val h1 = numericalDerivative(x, 0.1)
    val h2 = numericalDerivative(x, 0.01)
    val h3 = numericalDerivative(x, 0.001)
    val h4 = numericalDerivative(x, 0.0001)

    // As h shrinks, we converge to 0
    assertEqualsDouble(h1, 0.3, 0.01)
    assertEqualsDouble(h2, 0.03, 0.001)
    assertEqualsDouble(h3, 0.003, 0.0001)
    assertEqualsDouble(h4, 0.0003, 0.00001)

    // Very close to zero
    assert(math.abs(h4) < 0.001, "Derivative at x=2/3 should be near zero")
  }

  // ============================================================
  // Exploring partial derivatives: d = a * b + c
  //
  // For d = a * b + c:
  //   ∂d/∂a = b   (how much d changes when we nudge a)
  //   ∂d/∂b = a   (how much d changes when we nudge b)
  //   ∂d/∂c = 1   (how much d changes when we nudge c)
  // ============================================================

  val a = 2.0
  val b = -3.0
  val c = 10.0
  val h = 0.0001

  def d(a: Double, b: Double, c: Double): Double =
    (Value(a) * Value(b) + Value(c)).eval

  test("partial derivative ∂d/∂a = b (slope equals b)") {
    // d = a * b + c
    // When we bump a by h, d changes by approximately b * h
    // So (d(a+h) - d(a)) / h ≈ b
    val d1 = d(a, b, c)
    val d2 = d(a + h, b, c)
    val slope = (d2 - d1) / h

    // ∂d/∂a = b = -3.0
    assertEqualsDouble(slope, b, 0.0001)
    assert(slope < 0, "Slope should be negative since b is negative")
  }

  test("partial derivative ∂d/∂b = a (slope equals a)") {
    // d = a * b + c
    // When we bump b by h, d changes by approximately a * h
    // So (d(b+h) - d(b)) / h ≈ a
    val d1 = d(a, b, c)
    val d2 = d(a, b + h, c)
    val slope = (d2 - d1) / h

    // ∂d/∂b = a = 2.0
    assertEqualsDouble(slope, a, 0.0001)
    assert(slope > 0, "Slope should be positive since a is positive")
  }

  test("partial derivative ∂d/∂c = 1 (constant addition has slope 1)") {
    // d = a * b + c
    // When we bump c by h, d changes by exactly h
    // So (d(c+h) - d(c)) / h ≈ 1
    val d1 = d(a, b, c)
    val d2 = d(a, b, c + h)
    val slope = (d2 - d1) / h

    // ∂d/∂c = 1
    assertEqualsDouble(slope, 1.0, 0.0001)
  }

  test("sum of partial derivatives weighted by inputs") {
    // If we bump ALL inputs by the same h:
    // Δd ≈ (∂d/∂a)*h + (∂d/∂b)*h + (∂d/∂c)*h
    //    = b*h + a*h + 1*h
    //    = (a + b + 1)*h
    val d1 = d(a, b, c)
    val d2 = d(a + h, b + h, c + h)
    val totalChange = d2 - d1
    val expectedChange = (a + b + 1) * h

    assertEqualsDouble(totalChange, expectedChange, 0.0000001)
  }
