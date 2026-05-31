package palladium

class ValueSuite extends munit.FunSuite:

  test("Lit evaluates to its value") {
    val x = Value(42.0)
    assertEquals(x.eval, 42.0)
  }

  test("Const evaluates via fromInt") {
    val x = Value.const[Double](42)
    assertEquals(x.eval, 42.0)
  }

  test("Add evaluates correctly") {
    val x = Value(2.0)
    val y = Value(3.0)
    assertEquals((x + y).eval, 5.0)
  }

  test("Sub evaluates correctly") {
    val x = Value(5.0)
    val y = Value(3.0)
    assertEquals((x - y).eval, 2.0)
  }

  test("Mul evaluates correctly") {
    val x = Value(4.0)
    val y = Value(5.0)
    assertEquals((x * y).eval, 20.0)
  }

  test("Div evaluates correctly") {
    val x = Value(10.0)
    val y = Value(2.0)
    assertEquals((x / y).eval, 5.0)
  }

  test("FastDiv evaluates as a * b^(-1)") {
    val x = Value(10.0)
    val y = Value(2.0)
    // 10 * 2^(-1) = 10 * 0.5 = 5.0
    assertEquals((x /~ y).eval, 5.0)
  }

  test("FastDiv matches Div for simple values") {
    val x = Value(7.0)
    val y = Value(3.0)
    assertEqualsDouble((x /~ y).eval, (x / y).eval, 1e-10)
  }

  test("Pow with ~^ evaluates correctly") {
    val x = Value(2.0)
    val y = Value(3.0)
    assertEquals((x ~^ y).eval, 8.0)
  }

  test("~^ has higher precedence than * (a * b ~^ c = a * (b ~^ c))") {
    val a = Value(2.0)
    val b = Value(3.0)
    val c = Value(2.0)
    // ~^ is precedence 10 (highest), * is precedence 9
    // So a * b ~^ c should be a * (b^c) = 2 * 9 = 18, not (a*b)^c = 36
    assertEquals((a * b ~^ c).eval, 18.0)
    assertEquals((a * b ~^ c).eval, (a * (b ~^ c)).eval)
  }

  test("Pow with .pow evaluates correctly") {
    val x = Value(2.0)
    val y = Value(3.0)
    assertEquals(x.pow(y).eval, 8.0)
  }

  test("Neg evaluates correctly") {
    val x = Value(5.0)
    assertEquals((-x).eval, -5.0)
  }

  test("negate evaluates correctly") {
    val x = Value(5.0)
    assertEquals(x.negate.eval, -5.0)
  }

  test("negate is equivalent to unary_-") {
    val x = Value.variable("x", 3.0)
    assertEquals(x.negate.eval, (-x).eval)
  }

  test("compound expression: (2 + 3) * 4") {
    val result = (Value(2.0) + Value(3.0)) * Value(4.0)
    assertEquals(result.eval, 20.0)
  }

  test("compound expression: 10 / (2 + 3)") {
    val result = Value(10.0) / (Value(2.0) + Value(3.0))
    assertEquals(result.eval, 2.0)
  }

  test("nested expression: ((1 + 2) * (3 + 4)) / 7") {
    val a = Value(1.0) + Value(2.0)  // 3
    val b = Value(3.0) + Value(4.0)  // 7
    val c = (a * b) / Value(7.0)     // 21 / 7 = 3
    assertEquals(c.eval, 3.0)
  }

  test("expression tree structure is preserved") {
    val x = Value(2.0)
    val y = Value(3.0)
    val expr = x + y

    expr match
      case Value.Add(Value.Lit(l), Value.Lit(r)) =>
        assertEquals(l, 2.0)
        assertEquals(r, 3.0)
      case _ =>
        fail("Expected Add(Lit, Lit)")
  }

  test("works with Float") {
    val x = Value(2.0f)
    val y = Value(3.0f)
    assertEquals((x + y).eval, 5.0f)
  }

  // Polynomial: f(x) = 3x^2 - 4x + 5
  def f(x: Value[Double]): Value[Double] =
    Value.const(3) * (x ~^ Value.const(2)) - Value.const(4) * x + Value.const(5)

  test("polynomial f(x) = 3x^2 - 4x + 5 at x=0") {
    // f(0) = 0 - 0 + 5 = 5
    assertEquals(f(Value(0.0)).eval, 5.0)
  }

  test("polynomial f(x) = 3x^2 - 4x + 5 at x=1") {
    // f(1) = 3 - 4 + 5 = 4
    assertEquals(f(Value(1.0)).eval, 4.0)
  }

  test("polynomial f(x) = 3x^2 - 4x + 5 at x=2") {
    // f(2) = 12 - 8 + 5 = 9
    assertEquals(f(Value(2.0)).eval, 9.0)
  }

  test("polynomial f(x) = 3x^2 - 4x + 5 at x=-1") {
    // f(-1) = 3 + 4 + 5 = 12
    assertEquals(f(Value(-1.0)).eval, 12.0)
  }

  test("Exp evaluates to e^x") {
    val x = Value.variable("x", 1.0)
    assertEqualsDouble(x.exp.eval, math.E, 1e-10)
  }

  test("Exp at x=0 evaluates to 1.0") {
    val x = Value(0.0)
    assertEqualsDouble(x.exp.eval, 1.0, 1e-10)
  }

  test("Tanh at x=0 evaluates to 0.0") {
    val x = Value.variable("x", 0.0)
    assertEqualsDouble(x.tanh.eval, 0.0, 1e-10)
  }

  test("Tanh at x=1 evaluates to math.tanh(1)") {
    val x = Value.variable("x", 1.0)
    assertEqualsDouble(x.tanh.eval, math.tanh(1.0), 1e-10)
  }

  test("Tanh at x=-2 evaluates to math.tanh(-2)") {
    val x = Value(-2.0)
    assertEqualsDouble(x.tanh.eval, math.tanh(-2.0), 1e-10)
  }

  // ============================================
  // Sigmoid tests — σ(x) = 1 / (1 + exp(-x))
  // ============================================

  test("Sigmoid at x=0 evaluates to 0.5") {
    val x = Value(0.0)
    assertEqualsDouble(x.sigmoid.eval, 0.5, 1e-10)
  }

  test("Sigmoid at x=1") {
    val x = Value.variable("x", 1.0)
    val expected = 1.0 / (1.0 + math.exp(-1.0))
    assertEqualsDouble(x.sigmoid.eval, expected, 1e-10)
  }

  test("Sigmoid at large negative x is near 0") {
    val x = Value(-10.0)
    assert(x.sigmoid.eval < 0.001)
  }

  test("Sigmoid at large positive x is near 1") {
    val x = Value(10.0)
    assert(x.sigmoid.eval > 0.999)
  }

  // ============================================
  // ReLU tests — max(0, x)
  // ============================================

  test("ReLU at x=2 evaluates to 2") {
    val x = Value(2.0)
    assertEquals(x.relu.eval, 2.0)
  }

  test("ReLU at x=-3 evaluates to 0") {
    val x = Value(-3.0)
    assertEquals(x.relu.eval, 0.0)
  }

  test("ReLU at x=0 evaluates to 0") {
    val x = Value(0.0)
    assertEquals(x.relu.eval, 0.0)
  }

  // ============================================
  // Sin tests — sin(x)
  // ============================================

  test("Sin at x=0 evaluates to 0") {
    val x = Value(0.0)
    assertEqualsDouble(x.sin.eval, 0.0, 1e-10)
  }

  test("Sin at x=pi/2 evaluates to 1") {
    val x = Value(math.Pi / 2)
    assertEqualsDouble(x.sin.eval, 1.0, 1e-10)
  }

  test("Sin at x=pi evaluates to 0") {
    val x = Value(math.Pi)
    assertEqualsDouble(x.sin.eval, 0.0, 1e-10)
  }

  // ============================================
  // Cos tests — cos(x)
  // ============================================

  test("Cos at x=0 evaluates to 1") {
    val x = Value(0.0)
    assertEqualsDouble(x.cos.eval, 1.0, 1e-10)
  }

  test("Cos at x=pi/2 evaluates to 0") {
    val x = Value(math.Pi / 2)
    assertEqualsDouble(x.cos.eval, 0.0, 1e-10)
  }

  test("Cos at x=pi evaluates to -1") {
    val x = Value(math.Pi)
    assertEqualsDouble(x.cos.eval, -1.0, 1e-10)
  }

  // ============================================
  // Abs tests — |x|
  // ============================================

  test("Abs at x=3 evaluates to 3") {
    val x = Value(3.0)
    assertEquals(x.abs.eval, 3.0)
  }

  test("Abs at x=-5 evaluates to 5") {
    val x = Value(-5.0)
    assertEquals(x.abs.eval, 5.0)
  }

  test("Abs at x=0 evaluates to 0") {
    val x = Value(0.0)
    assertEquals(x.abs.eval, 0.0)
  }

  test("polynomial expression tree uses Const") {
    val expr = f(Value(2.0))

    // The tree should contain Const nodes
    def containsConst(v: Value[Double]): Boolean = v match
      case Value.Const(_)       => true
      case Value.Add(l, r)      => containsConst(l) || containsConst(r)
      case Value.Sub(l, r)      => containsConst(l) || containsConst(r)
      case Value.Mul(l, r)      => containsConst(l) || containsConst(r)
      case Value.Div(l, r)      => containsConst(l) || containsConst(r)
      case Value.Pow(b, e)      => containsConst(b) || containsConst(e)
      case Value.Neg(v)         => containsConst(v)
      case Value.Log(v)         => containsConst(v)
      case Value.Exp(v)         => containsConst(v)
      case Value.Tanh(v)        => containsConst(v)
      case Value.Sigmoid(v)     => containsConst(v)
      case Value.Relu(v)        => containsConst(v)
      case Value.Sin(v)         => containsConst(v)
      case Value.Cos(v)         => containsConst(v)
      case Value.Abs(v)         => containsConst(v)
      case Value.Var(_, _)      => false
      case Value.Lit(_)         => false
      case Value.FastDiv(l, r)  => containsConst(l) || containsConst(r)

    assert(containsConst(expr), "Expression should contain Const nodes")
  }
