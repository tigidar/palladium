package palladium.blas

import palladium.ein.*

/** Tests for BlasGen — the high-level Ein → BLAS evaluation API.
  * Verifies the full pipeline: Ein → Lower → BlasExec → TensorData.
  */
class BlasGenSuite extends munit.FunSuite:

  val eps = 1e-8

  test("BlasGen.eval: simple matrix-vector multiply") {
    val inp = Dim("inp", 2)
    val hid = Dim("hid", 3)
    val wData = TensorData.fromArray(List(hid, inp), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val w = Ein.Param[Double]("W", List(hid, inp), wData)
    val x = Ein.Input[Double]("x", List(inp))
    val expr = w * x

    val feed = Map("x" -> TensorData.fromArray(List(inp), Array(1.0, 2.0)))
    val expected = EinEval.eval(expr, feed)
    val actual = BlasGen.eval(expr, feed)

    var i = 0
    while i < expected.data.length do
      assertEqualsDouble(actual.data(i), expected.data(i), eps)
      i += 1
  }

  test("BlasGen.eval: 2-layer MLP end-to-end") {
    val inp = Dim("inp", 2)
    val hid = Dim("hid", 4)
    val out = Dim("out", 1)

    val w1 = Ein.Param[Double]("W1", List(hid, inp),
      TensorData.fromArray(List(hid, inp), Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)))
    val b1 = Ein.Param[Double]("b1", List(hid),
      TensorData.fromArray(List(hid), Array(0.01, 0.02, 0.03, 0.04)))
    val w2 = Ein.Param[Double]("W2", List(out, hid),
      TensorData.fromArray(List(out, hid), Array(0.1, 0.2, 0.3, 0.4)))
    val b2 = Ein.Param[Double]("b2", List(out),
      TensorData.fromArray(List(out), Array(0.05)))
    val x = Ein.Input[Double]("x", List(inp))

    val h = Ein.Activate(Activation.ReLU, (w1 * x) + b1)
    val output = (w2 * h) + b2

    val feed = Map("x" -> TensorData.fromArray(List(inp), Array(1.0, 2.0)))
    val expected = EinEval.eval(output, feed)
    val actual = BlasGen.eval(output, feed)

    var i = 0
    while i < expected.data.length do
      assertEqualsDouble(actual.data(i), expected.data(i), eps)
      i += 1
  }

  test("BlasGen.eval: Block DSL network") {
    val inp = Dim("inp", 3)
    val input = Ein.Input[Double]("x", List(inp))
    val net = (Block.dense[Double](4, Activation.Sigmoid) >> Block.linear[Double](2))
      .materialize(input)

    val feed = Map("x" -> TensorData.fromArray(List(inp), Array(0.5, -0.3, 0.8)))
    val expected = EinEval.eval(net, feed)
    val actual = BlasGen.eval(net, feed)

    var i = 0
    while i < expected.data.length do
      assertEqualsDouble(actual.data(i), expected.data(i), eps)
      i += 1
  }

  test("BlasGen.evalRaw returns raw arrays") {
    val d = Dim("d", 3)
    val x = Ein.Input[Double]("x", List(d))
    val expr = Ein.Activate(Activation.Tanh, x)
    val feed = Map("x" -> TensorData.fromArray(List(d), Array(-1.0, 0.0, 1.0)))

    val results = BlasGen.evalRaw(expr, feed)
    assertEquals(results.length, 1)
    assertEquals(results.head.length, 3)
    assertEqualsDouble(results.head(1), 0.0, eps) // tanh(0) = 0
  }
