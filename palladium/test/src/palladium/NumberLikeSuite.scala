package palladium

class NumberLikeSuite extends munit.FunSuite:

  val doubleTol = 1e-10
  val floatTol = 1e-5f

  // ============================================
  // Double instance
  // ============================================

  test("Double: basic arithmetic") {
    val num = summon[NumberLike[Double]]
    assertEqualsDouble(num.plus(2.0, 3.0), 5.0, doubleTol)
    assertEqualsDouble(num.minus(5.0, 3.0), 2.0, doubleTol)
    assertEqualsDouble(num.times(4.0, 3.0), 12.0, doubleTol)
    assertEqualsDouble(num.div(10.0, 4.0), 2.5, doubleTol)
  }

  test("Double: pow and sqrt") {
    val num = summon[NumberLike[Double]]
    assertEqualsDouble(num.pow(2.0, 3.0), 8.0, doubleTol)
    assertEqualsDouble(num.sqrt(9.0), 3.0, doubleTol)
  }

  test("Double: transcendentals") {
    val num = summon[NumberLike[Double]]
    assertEqualsDouble(num.exp(0.0), 1.0, doubleTol)
    assertEqualsDouble(num.log(1.0), 0.0, doubleTol)
    assertEqualsDouble(num.sin(0.0), 0.0, doubleTol)
    assertEqualsDouble(num.cos(0.0), 1.0, doubleTol)
  }

  test("Double: abs, signum, negate") {
    val num = summon[NumberLike[Double]]
    assertEqualsDouble(num.abs(-5.0), 5.0, doubleTol)
    assertEqualsDouble(num.abs(3.0), 3.0, doubleTol)
    assertEqualsDouble(num.signum(-5.0), -1.0, doubleTol)
    assertEqualsDouble(num.signum(0.0), 0.0, doubleTol)
    assertEqualsDouble(num.signum(5.0), 1.0, doubleTol)
    assertEqualsDouble(num.negate(3.0), -3.0, doubleTol)
  }

  test("Double: relu and step") {
    val num = summon[NumberLike[Double]]
    assertEqualsDouble(num.relu(-2.0), 0.0, doubleTol)
    assertEqualsDouble(num.relu(3.0), 3.0, doubleTol)
    assertEqualsDouble(num.relu(0.0), 0.0, doubleTol)
    assertEqualsDouble(num.step(-2.0), 0.0, doubleTol)
    assertEqualsDouble(num.step(3.0), 1.0, doubleTol)
    assertEqualsDouble(num.step(0.0), 0.0, doubleTol)
  }

  test("Double: max and negativeInfinity") {
    val num = summon[NumberLike[Double]]
    assertEqualsDouble(num.max(3.0, 5.0), 5.0, doubleTol)
    assertEqualsDouble(num.max(-1.0, -5.0), -1.0, doubleTol)
    assert(num.negativeInfinity < -1e100)
  }

  test("Double: fromInt") {
    val num = summon[NumberLike[Double]]
    assertEqualsDouble(num.fromInt(0), 0.0, doubleTol)
    assertEqualsDouble(num.fromInt(42), 42.0, doubleTol)
    assertEqualsDouble(num.fromInt(-7), -7.0, doubleTol)
  }

  // ============================================
  // Float instance
  // ============================================

  test("Float: basic arithmetic") {
    val num = summon[NumberLike[Float]]
    assertEqualsFloat(num.plus(2.0f, 3.0f), 5.0f, floatTol)
    assertEqualsFloat(num.minus(5.0f, 3.0f), 2.0f, floatTol)
    assertEqualsFloat(num.times(4.0f, 3.0f), 12.0f, floatTol)
    assertEqualsFloat(num.div(10.0f, 4.0f), 2.5f, floatTol)
  }

  test("Float: pow and sqrt") {
    val num = summon[NumberLike[Float]]
    assertEqualsFloat(num.pow(2.0f, 3.0f), 8.0f, floatTol)
    assertEqualsFloat(num.sqrt(9.0f), 3.0f, floatTol)
  }

  test("Float: transcendentals") {
    val num = summon[NumberLike[Float]]
    assertEqualsFloat(num.exp(0.0f), 1.0f, floatTol)
    assertEqualsFloat(num.log(1.0f), 0.0f, floatTol)
    assertEqualsFloat(num.sin(0.0f), 0.0f, floatTol)
    assertEqualsFloat(num.cos(0.0f), 1.0f, floatTol)
  }

  test("Float: abs, signum, negate") {
    val num = summon[NumberLike[Float]]
    assertEqualsFloat(num.abs(-5.0f), 5.0f, floatTol)
    assertEqualsFloat(num.signum(-5.0f), -1.0f, floatTol)
    assertEqualsFloat(num.signum(0.0f), 0.0f, floatTol)
    assertEqualsFloat(num.negate(3.0f), -3.0f, floatTol)
  }

  test("Float: relu and step") {
    val num = summon[NumberLike[Float]]
    assertEqualsFloat(num.relu(-2.0f), 0.0f, floatTol)
    assertEqualsFloat(num.relu(3.0f), 3.0f, floatTol)
    assertEqualsFloat(num.step(-2.0f), 0.0f, floatTol)
    assertEqualsFloat(num.step(3.0f), 1.0f, floatTol)
  }

  test("Float: max and negativeInfinity") {
    val num = summon[NumberLike[Float]]
    assertEqualsFloat(num.max(3.0f, 5.0f), 5.0f, floatTol)
    assert(num.negativeInfinity < -1e30f)
  }

  test("Float: fromInt") {
    val num = summon[NumberLike[Float]]
    assertEqualsFloat(num.fromInt(42), 42.0f, floatTol)
  }
