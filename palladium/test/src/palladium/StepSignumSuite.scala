package palladium

import munit.FunSuite
import Value.*

class StepSignumSuite extends FunSuite:

  // -- Step --

  test("step(x) for x > 0 evaluates to 1"):
    val x = Var("x", 3.0)
    assertEquals(x.step.eval, 1.0)

  test("step(x) for x < 0 evaluates to 0"):
    val x = Var("x", -2.0)
    assertEquals(x.step.eval, 0.0)

  test("step(x) for x = 0 evaluates to 0"):
    val x = Var("x", 0.0)
    assertEquals(x.step.eval, 0.0)

  // -- Signum --

  test("signum(x) for x > 0 evaluates to 1"):
    val x = Var("x", 5.0)
    assertEquals(x.signum.eval, 1.0)

  test("signum(x) for x < 0 evaluates to -1"):
    val x = Var("x", -3.0)
    assertEquals(x.signum.eval, -1.0)

  test("signum(x) for x = 0 evaluates to 0"):
    val x = Var("x", 0.0)
    assertEquals(x.signum.eval, 0.0)

  // -- Symbolic gradients safe at zero --

  test("symbolic gradient of relu(x) at x = 0 does not produce NaN"):
    import SymbolicGrad.syntax.*
    val x = Var("x", 0.0)
    val grads = x.relu.symbolicBackward
    val gradVal = grads("x").eval
    assert(!gradVal.isNaN, s"Gradient at relu(0) should not be NaN, got $gradVal")
    assertEquals(gradVal, 0.0)

  test("symbolic gradient of abs(x) at x = 0 does not produce NaN"):
    import SymbolicGrad.syntax.*
    val x = Var("x", 0.0)
    val grads = x.abs.symbolicBackward
    val gradVal = grads("x").eval
    assert(!gradVal.isNaN, s"Gradient at abs(0) should not be NaN, got $gradVal")

  test("symbolic gradient of relu(x) at x > 0 matches numerical"):
    import SymbolicGrad.syntax.*
    val x = Var("x", 2.0)
    val symbolicGrads = x.relu.symbolicBackward
    val numericalGrads = Grad.backward(x.relu)
    assertEqualsDouble(symbolicGrads("x").eval, numericalGrads("x"), 1e-10)

  test("symbolic gradient of relu(x) at x < 0 matches numerical"):
    import SymbolicGrad.syntax.*
    val x = Var("x", -2.0)
    val symbolicGrads = x.relu.symbolicBackward
    val numericalGrads = Grad.backward(x.relu)
    assertEqualsDouble(symbolicGrads("x").eval, numericalGrads("x"), 1e-10)

  // -- Step/Signum are piecewise constant, gradient is zero --

  test("numerical gradient of step is zero"):
    val x = Var("x", 2.0)
    val grads = Grad.backward(x.step)
    assertEquals(grads.getOrElse("x", 0.0), 0.0)

  test("numerical gradient of signum is zero"):
    val x = Var("x", 2.0)
    val grads = Grad.backward(x.signum)
    assertEquals(grads.getOrElse("x", 0.0), 0.0)
