package palladium

import palladium.Dsl.{*, given}

class DslSuite extends munit.FunSuite:

  test("toVar creates Value.Var with variable name") {
    val myVar = 5.0.toVar
    assertEquals(myVar, Value.Var("myVar", 5.0))
  }

  test("^ creates Value.Var with variable name") {
    val x = 10.0.^
    assertEquals(x, Value.Var("x", 10.0))
  }

  test("^ works with Float") {
    val f = 3.14f.^
    assertEquals(f, Value.Var("f", 3.14f))
  }

  test("implicit conversion allows x + 2.0") {
    val x = 1.0.^
    val expr = x + 2.0
    assertEquals(expr.eval, 3.0)
  }

  test("implicit conversion allows 2.0 * x") {
    val x = 3.0.^
    val expr: Value[Double] = 2.0 * x
    assertEquals(expr.eval, 6.0)
  }

  test("compound expression with implicit conversions") {
    val x = 2.0.^
    val y = 3.0.^
    val expr = x * x + 2.0 * y
    assertEquals(expr.eval, 10.0)
  }

  test("expression tree uses Var nodes from ^") {
    val x = 5.0.^
    x match
      case Value.Var(id, data) =>
        assertEquals(id, "x")
        assertEquals(data, 5.0)
      case _ =>
        fail("Expected Value.Var")
  }

  test("multiple variables in expression") {
    val a = 1.0.^
    val b = 2.0.^
    val c = 3.0.^
    val expr = a + b * c
    assertEquals(expr.eval, 7.0)
  }
