package palladium

import palladium.*
import palladium.Dsl.{*, given}

class ManualValueSuite extends munit.FunSuite:

  test("a * b + c") {
    val a = 2.0.^
    val b = -3.0.^
    val c = 10.0.^
    val expr = a * b + c
    assertEquals(expr.eval, 4.0)
  }

  test("x1*w1 + x2*w2 + b") {
    val x1 = 2.0.^
    val x2 = 0.0.^

    val w1 = -3.0.^
    val w2 = 1.0.^

    val b = 6.8813735870195432.^

    val L = tanh(x1 * w1 + x2 * w2 + b).eval

    assertEquals(L.eval, 0.7071067811865477)
  }
