package palladium

class GradientSuite extends munit.FunSuite:

  val epsilon = 1e-7
  val tolerance = 1e-5

  // Helper for numerical gradient verification
  def numericalGradient(
      f: Double => Double,
      x: Double,
      h: Double = 1e-7
  ): Double =
    (f(x + h) - f(x - h)) / (2 * h)

  // ============================================
  // Basic gradient tests
  // ============================================

  test("single variable: x -> grad 1.0") {
    val x = Value.variable("x", 2.0)
    val grads = Grad.backward(x)
    assertEquals(grads("x"), 1.0)
  }

  test("literal has no gradient") {
    val lit = Value(5.0)
    val grads = Grad.backward(lit)
    assertEquals(grads.size, 0)
  }

  test("const has no gradient") {
    val c = Value.const[Double](5)
    val grads = Grad.backward(c)
    assertEquals(grads.size, 0)
  }

  // ============================================
  // Addition tests
  // ============================================

  test("addition: x + y -> grads (1.0, 1.0)") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val expr = x + y
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), 1.0)
    assertEquals(grads("y"), 1.0)
  }

  test("addition with literal: x + 5 -> grad 1.0") {
    val x = Value.variable("x", 2.0)
    val expr = x + Value(5.0)
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), 1.0)
  }

  // ============================================
  // Subtraction tests
  // ============================================

  test("subtraction: x - y -> grads (1.0, -1.0)") {
    val x = Value.variable("x", 5.0)
    val y = Value.variable("y", 3.0)
    val expr = x - y
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), 1.0)
    assertEquals(grads("y"), -1.0)
  }

  // ============================================
  // Multiplication tests
  // ============================================

  test("multiplication: x * y -> grads (y, x)") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val expr = x * y
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), 3.0) // grad w.r.t x = y
    assertEquals(grads("y"), 2.0) // grad w.r.t y = x
  }

  test("multiplication with literal: x * 3 -> grad 3.0") {
    val x = Value.variable("x", 2.0)
    val expr = x * Value(3.0)
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), 3.0)
  }

  // ============================================
  // Shared variable (gradient accumulation)
  // ============================================

  test("shared variable: x * x -> grad 2x") {
    val x = Value.variable("x", 3.0)
    val expr = x * x
    val grads = Grad.backward(expr)
    // d/dx (x^2) = 2x = 6
    assertEquals(grads("x"), 6.0)
  }

  test("shared variable: x + x -> grad 2.0") {
    val x = Value.variable("x", 5.0)
    val expr = x + x
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), 2.0)
  }

  test("shared variable: x * x + x -> grad 2x + 1") {
    val x = Value.variable("x", 2.0)
    val expr = x * x + x
    val grads = Grad.backward(expr)
    // d/dx (x^2 + x) = 2x + 1 = 5
    assertEquals(grads("x"), 5.0)
  }

  // ============================================
  // Division tests
  // ============================================

  test("division: x / y") {
    val x = Value.variable("x", 6.0)
    val y = Value.variable("y", 2.0)
    val expr = x / y
    val grads = Grad.backward(expr)
    // d/dx (x/y) = 1/y = 0.5
    // d/dy (x/y) = -x/y^2 = -6/4 = -1.5
    assertEqualsDouble(grads("x"), 0.5, tolerance)
    assertEqualsDouble(grads("y"), -1.5, tolerance)
  }

  test("division: 1/x") {
    val x = Value.variable("x", 2.0)
    val expr = Value(1.0) / x
    val grads = Grad.backward(expr)
    // d/dx (1/x) = -1/x^2 = -0.25
    assertEqualsDouble(grads("x"), -0.25, tolerance)
  }

  // ============================================
  // Power tests
  // ============================================

  test("power: x^2") {
    val x = Value.variable("x", 3.0)
    val expr = x ~^ Value(2.0)
    val grads = Grad.backward(expr)
    // d/dx (x^2) = 2x = 6
    assertEqualsDouble(grads("x"), 6.0, tolerance)
  }

  test("power: x^3") {
    val x = Value.variable("x", 2.0)
    val expr = x ~^ Value(3.0)
    val grads = Grad.backward(expr)
    // d/dx (x^3) = 3x^2 = 12
    assertEqualsDouble(grads("x"), 12.0, tolerance)
  }

  test("power with variable exponent: 2^x") {
    val x = Value.variable("x", 3.0)
    val expr = Value(2.0) ~^ x
    val grads = Grad.backward(expr)
    // d/dx (2^x) = 2^x * ln(2) = 8 * ln(2)
    val expected = 8.0 * math.log(2.0)
    assertEqualsDouble(grads("x"), expected, tolerance)
  }

  test("power with both variable: x^y") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val expr = x ~^ y
    val grads = Grad.backward(expr)
    // d/dx (x^y) = y * x^(y-1) = 3 * 4 = 12
    // d/dy (x^y) = x^y * ln(x) = 8 * ln(2)
    assertEqualsDouble(grads("x"), 12.0, tolerance)
    assertEqualsDouble(grads("y"), 8.0 * math.log(2.0), tolerance)
  }

  // ============================================
  // Negation tests
  // ============================================

  test("negation: -x") {
    val x = Value.variable("x", 5.0)
    val expr = -x
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), -1.0)
  }

  test("negate method: gradient matches unary_-") {
    val x = Value.variable("x", 2.0)
    val expected = Grad.backward(-x)
    val actual = Grad.backward(x.negate)
    assertEquals(actual("x"), expected("x"))
  }

  test("negation in expression: -x + y") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val expr = -x + y
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), -1.0)
    assertEquals(grads("y"), 1.0)
  }

  // ============================================
  // Polynomial tests
  // ============================================

  test("polynomial: 3x^2 - 4x + 5 at x=2") {
    val x = Value.variable("x", 2.0)
    val expr =
      Value.const(3) * (x ~^ Value.const(2)) - Value.const(4) * x + Value
        .const(5)
    val grads = Grad.backward(expr)
    // f(x) = 3x^2 - 4x + 5
    // f'(x) = 6x - 4 = 12 - 4 = 8 at x=2
    assertEqualsDouble(grads("x"), 8.0, tolerance)
  }

  test("polynomial: 3x^2 - 4x + 5 at x=0") {
    val x = Value.variable("x", 0.0)
    val expr =
      Value.const(3) * (x ~^ Value.const(2)) - Value.const(4) * x + Value
        .const(5)
    val grads = Grad.backward(expr)
    // f'(x) = 6x - 4 = -4 at x=0
    assertEqualsDouble(grads("x"), -4.0, tolerance)
  }

  test("polynomial: 3x^2 - 4x + 5 at x=1") {
    val x = Value.variable("x", 1.0)
    val expr =
      Value.const(3) * (x ~^ Value.const(2)) - Value.const(4) * x + Value
        .const(5)
    val grads = Grad.backward(expr)
    // f'(x) = 6x - 4 = 2 at x=1
    assertEqualsDouble(grads("x"), 2.0, tolerance)
  }

  // ============================================
  // Multi-variable expressions
  // ============================================

  test("multi-variable: x*y + x*z") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val z = Value.variable("z", 4.0)
    val expr = x * y + x * z
    val grads = Grad.backward(expr)
    // d/dx = y + z = 7
    // d/dy = x = 2
    // d/dz = x = 2
    assertEqualsDouble(grads("x"), 7.0, tolerance)
    assertEqualsDouble(grads("y"), 2.0, tolerance)
    assertEqualsDouble(grads("z"), 2.0, tolerance)
  }

  // ============================================
  // Numerical gradient verification
  // ============================================

  test("numerical verification: x^2") {
    val xVal = 3.0
    def f(x: Double): Double = x * x
    val x = Value.variable("x", xVal)
    val expr = x * x
    val computed = Grad.backward(expr)("x")
    val numerical = numericalGradient(f, xVal)
    assertEqualsDouble(computed, numerical, tolerance)
  }

  test("numerical verification: x^3 + 2x^2 - x + 1") {
    val xVal = 2.5
    def f(x: Double): Double = x * x * x + 2 * x * x - x + 1
    val x = Value.variable("x", xVal)
    val expr = (x ~^ Value(3.0)) + Value(2.0) * (x ~^ Value(2.0)) - x + Value(
      1.0
    )
    val computed = Grad.backward(expr)("x")
    // f'(x) = 3x^2 + 4x - 1
    val numerical = numericalGradient(f, xVal)
    assertEqualsDouble(computed, numerical, tolerance)
  }

  test("numerical verification: x / (1 + x^2)") {
    val xVal = 1.5
    def f(x: Double): Double = x / (1 + x * x)
    val x = Value.variable("x", xVal)
    val one = Value(1.0)
    val expr = x / (one + x * x)
    val computed = Grad.backward(expr)("x")
    val numerical = numericalGradient(f, xVal)
    assertEqualsDouble(computed, numerical, tolerance)
  }

  // ============================================
  // Fast division tests
  // ============================================

  test("fast division: x /~ y matches regular division gradients") {
    val x = Value.variable("x", 6.0)
    val y = Value.variable("y", 2.0)
    val fastExpr = x /~ y
    val divExpr = x / y
    val fastGrads = Grad.backward(fastExpr)
    val divGrads = Grad.backward(divExpr)
    // Gradients should match (same math, different computation path)
    assertEqualsDouble(fastGrads("x"), divGrads("x"), tolerance)
    assertEqualsDouble(fastGrads("y"), divGrads("y"), tolerance)
  }

  test("fast division: numerical verification x /~ y") {
    val xVal = 6.0
    val yVal = 3.0
    val x = Value.variable("x", xVal)
    val y = Value.variable("y", yVal)
    val expr = x /~ y
    val grads = Grad.backward(expr)
    // d/dx (x * y^(-1)) = y^(-1) = 1/3
    assertEqualsDouble(grads("x"), 1.0 / yVal, tolerance)
    // d/dy (x * y^(-1)) = -x * y^(-2) = -6/9 = -2/3
    assertEqualsDouble(grads("y"), -xVal / (yVal * yVal), tolerance)
  }

  test("numerical verification: x /~ (1 + x^2)") {
    val xVal = 1.5
    def f(x: Double): Double = x * math.pow(1 + x * x, -1)
    val x = Value.variable("x", xVal)
    val one = Value(1.0)
    val expr = x /~ (one + x * x)
    val computed = Grad.backward(expr)("x")
    val numerical = numericalGradient(f, xVal)
    assertEqualsDouble(computed, numerical, tolerance)
  }

  // ============================================
  // Exp tests
  // ============================================

  test("exp: d/dx exp(x) = exp(x)") {
    val x = Value.variable("x", 1.0)
    val expr = x.exp
    val grads = Grad.backward(expr)
    assertEqualsDouble(grads("x"), math.exp(1.0), tolerance)
  }

  test("numerical verification: exp(x)") {
    val xVal = 2.0
    def f(x: Double): Double = math.exp(x)
    val x = Value.variable("x", xVal)
    val expr = x.exp
    val computed = Grad.backward(expr)("x")
    val numerical = numericalGradient(f, xVal)
    assertEqualsDouble(computed, numerical, tolerance)
  }

  // ============================================
  // Tanh tests
  // ============================================

  test("tanh: d/dx tanh(x) = 1 - tanh(x)^2") {
    val xVal = 1.0
    val x = Value.variable("x", xVal)
    val expr = x.tanh
    val grads = Grad.backward(expr)
    val expected = 1.0 - math.tanh(xVal) * math.tanh(xVal)
    assertEqualsDouble(grads("x"), expected, tolerance)
  }

  test("numerical verification: tanh(x)") {
    val xVal = 0.5
    def f(x: Double): Double = math.tanh(x)
    val x = Value.variable("x", xVal)
    val expr = x.tanh
    val computed = Grad.backward(expr)("x")
    val numerical = numericalGradient(f, xVal)
    assertEqualsDouble(computed, numerical, tolerance)
  }

  // ============================================
  // Sigmoid tests — σ(x) = 1/(1+exp(-x)), σ'(x) = σ(x)(1-σ(x))
  // ============================================

  test("sigmoid: d/dx σ(x) = σ(x)(1-σ(x))") {
    val xVal = 1.0
    val x = Value.variable("x", xVal)
    val expr = x.sigmoid
    val grads = Grad.backward(expr)
    val s = 1.0 / (1.0 + math.exp(-xVal))
    val expected = s * (1.0 - s)
    assertEqualsDouble(grads("x"), expected, tolerance)
  }

  test("numerical verification: sigmoid(x)") {
    val xVal = 0.5
    def f(x: Double): Double = 1.0 / (1.0 + math.exp(-x))
    val x = Value.variable("x", xVal)
    val expr = x.sigmoid
    val computed = Grad.backward(expr)("x")
    val numerical = numericalGradient(f, xVal)
    assertEqualsDouble(computed, numerical, tolerance)
  }

  test("sigmoid: gradient in compound expression x * sigmoid(y)") {
    val xVal = 2.0
    val yVal = 1.0
    val x = Value.variable("x", xVal)
    val y = Value.variable("y", yVal)
    val expr = x * y.sigmoid
    val grads = Grad.backward(expr)
    val s = 1.0 / (1.0 + math.exp(-yVal))
    // d/dx = σ(y)
    assertEqualsDouble(grads("x"), s, tolerance)
    // d/dy = x * σ(y)(1-σ(y))
    assertEqualsDouble(grads("y"), xVal * s * (1.0 - s), tolerance)
  }

  // ============================================
  // ReLU tests — relu(x) = max(0,x), relu'(x) = step(x)
  // ============================================

  test("relu: d/dx relu(x) = 1 for x > 0") {
    val x = Value.variable("x", 3.0)
    val expr = x.relu
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), 1.0)
  }

  test("relu: d/dx relu(x) = 0 for x < 0") {
    val x = Value.variable("x", -2.0)
    val expr = x.relu
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), 0.0)
  }

  test("relu: d/dx relu(x) = 0 at x = 0") {
    val x = Value.variable("x", 0.0)
    val expr = x.relu
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), 0.0)
  }

  test("numerical verification: relu(x) for x > 0") {
    val xVal = 2.0
    def f(x: Double): Double = math.max(0.0, x)
    val x = Value.variable("x", xVal)
    val expr = x.relu
    val computed = Grad.backward(expr)("x")
    val numerical = numericalGradient(f, xVal)
    assertEqualsDouble(computed, numerical, tolerance)
  }

  // ============================================
  // Sin tests — d/dx sin(x) = cos(x)
  // ============================================

  test("sin: d/dx sin(x) = cos(x)") {
    val xVal = 1.0
    val x = Value.variable("x", xVal)
    val expr = x.sin
    val grads = Grad.backward(expr)
    assertEqualsDouble(grads("x"), math.cos(xVal), tolerance)
  }

  test("numerical verification: sin(x)") {
    val xVal = 2.0
    def f(x: Double): Double = math.sin(x)
    val x = Value.variable("x", xVal)
    val expr = x.sin
    val computed = Grad.backward(expr)("x")
    val numerical = numericalGradient(f, xVal)
    assertEqualsDouble(computed, numerical, tolerance)
  }

  // ============================================
  // Cos tests — d/dx cos(x) = -sin(x)
  // ============================================

  test("cos: d/dx cos(x) = -sin(x)") {
    val xVal = 1.0
    val x = Value.variable("x", xVal)
    val expr = x.cos
    val grads = Grad.backward(expr)
    assertEqualsDouble(grads("x"), -math.sin(xVal), tolerance)
  }

  test("numerical verification: cos(x)") {
    val xVal = 2.0
    def f(x: Double): Double = math.cos(x)
    val x = Value.variable("x", xVal)
    val expr = x.cos
    val computed = Grad.backward(expr)("x")
    val numerical = numericalGradient(f, xVal)
    assertEqualsDouble(computed, numerical, tolerance)
  }

  // ============================================
  // Abs tests — d/dx |x| = sign(x)
  // ============================================

  test("abs: d/dx |x| = 1 for x > 0") {
    val x = Value.variable("x", 3.0)
    val expr = x.abs
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), 1.0)
  }

  test("abs: d/dx |x| = -1 for x < 0") {
    val x = Value.variable("x", -3.0)
    val expr = x.abs
    val grads = Grad.backward(expr)
    assertEquals(grads("x"), -1.0)
  }

  test("numerical verification: abs(x) for x > 0") {
    val xVal = 2.0
    def f(x: Double): Double = math.abs(x)
    val x = Value.variable("x", xVal)
    val expr = x.abs
    val computed = Grad.backward(expr)("x")
    val numerical = numericalGradient(f, xVal)
    assertEqualsDouble(computed, numerical, tolerance)
  }

  test("numerical verification: abs(x) for x < 0") {
    val xVal = -2.0
    def f(x: Double): Double = math.abs(x)
    val x = Value.variable("x", xVal)
    val expr = x.abs
    val computed = Grad.backward(expr)("x")
    val numerical = numericalGradient(f, xVal)
    assertEqualsDouble(computed, numerical, tolerance)
  }

  // ============================================

  test("Grad.syntax extension method") {
    import Grad.syntax.*
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val expr = x * y
    val grads = expr.backward
    assertEquals(grads("x"), 3.0)
    assertEquals(grads("y"), 2.0)
  }

  // ============================================
  // AutoGrad tests
  // ============================================

  test("AutoGrad.trace returns value and grads") {
    val x = Value.variable("x", 2.0)
    val expr = x * x + x
    val result = AutoGrad.trace(expr)
    // value = 4 + 2 = 6
    assertEquals(result.value, 6.0)
    // grad = 2x + 1 = 5
    assertEquals(result.grads("x"), 5.0)
  }

  test("AutoGrad.syntax extension method") {
    import AutoGrad.syntax.*
    val x = Value.variable("x", 3.0)
    val result = (x ~^ Value(2.0)).trace
    assertEquals(result.value, 9.0)
    assertEqualsDouble(result.grads("x"), 6.0, tolerance)
  }

  test("Traced case class destructuring") {
    val x = Value.variable("x", 2.0)
    val Traced(value, grads) = AutoGrad.trace(x * x)
    assertEquals(value, 4.0)
    assertEquals(grads("x"), 4.0)
  }
