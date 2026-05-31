package palladium.blas

import palladium.codegen.*
import palladium.ein.*
import scala.scalanative.unsafe.*

/** Tests for BlasExec — the BLAS-backed LowProgram interpreter.
  *
  * Strategy: build Ein expressions, lower to LowProgram, execute via
  * BlasExec, and compare results against EinEval (the reference interpreter).
  */
class BlasExecSuite extends munit.FunSuite:

  val eps = 1e-8

  // ── Helper: compare BlasExec result against EinEval ─────────────────

  private def assertBlasMatchesEval(
      expr: Ein[Double],
      feed: Map[String, TensorData[Double]],
      tolerance: Double = eps
  ): Unit =
    val expected = EinEval.eval(expr, feed)
    val prog = Lower.lower(expr)

    // Build param data map (name → Array[Double])
    val paramData: Map[String, Array[Double]] = prog.params.map { (ref, name) =>
      expr match
        case _ =>
          val td = collectParams(expr)
          name -> td.getOrElse(name, Array.emptyDoubleArray)
    }.toMap

    // Build input data map
    val inputData: Map[String, Array[Double]] = feed.map { (name, td) =>
      name -> td.data.map(_.toDouble)
    }

    val result = BlasExec.run(prog, paramData, inputData)
    val actual = result.head

    assertEquals(actual.length, expected.data.length,
      s"Output size mismatch: got ${actual.length}, expected ${expected.data.length}")

    var i = 0
    while i < actual.length do
      assertEqualsDouble(actual(i), expected.data(i), tolerance,
        s"Mismatch at index $i: got ${actual(i)}, expected ${expected.data(i)}")
      i += 1

  /** Recursively collect param data from an Ein expression tree. */
  private def collectParams(expr: Ein[Double]): Map[String, Array[Double]] =
    expr match
      case Ein.Param(id, _, data) => Map(id -> data.data.map(_.toDouble))
      case Ein.Contract(l, r)     => collectParams(l) ++ collectParams(r)
      case Ein.ElemAdd(l, r)      => collectParams(l) ++ collectParams(r)
      case Ein.ElemSub(l, r)      => collectParams(l) ++ collectParams(r)
      case Ein.ElemMul(l, r)      => collectParams(l) ++ collectParams(r)
      case Ein.Activate(_, arg)   => collectParams(arg)
      case Ein.ActivateDeriv(_, arg) => collectParams(arg)
      case Ein.ReduceSum(arg, _)  => collectParams(arg)
      case Ein.Broadcast(arg, _)  => collectParams(arg)
      case Ein.Transpose(arg, _)  => collectParams(arg)
      case Ein.Softmax(arg, _)    => collectParams(arg)
      case Ein.LogSoftmax(arg, _) => collectParams(arg)
      case Ein.LayerNorm(arg, s, b, _, _) =>
        collectParams(arg) ++ collectParams(s) ++ collectParams(b)
      case Ein.Reshape(arg, _)    => collectParams(arg)
      case Ein.Slice(arg, _, _, _) => collectParams(arg)
      case Ein.Gather(t, _, _)    => collectParams(t)
      case Ein.Scatter(s, _, _, _) => collectParams(s)
      case _                      => Map.empty

  // ── MatMul tests ────────────────────────────────────────────────────

  test("matrix-vector multiply: W * x") {
    val inp = Dim("inp", 2)
    val hid = Dim("hid", 3)
    val wData = TensorData.fromArray(List(hid, inp), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val w = Ein.Param[Double]("W", List(hid, inp), wData)
    val x = Ein.Input[Double]("x", List(inp))
    val expr = w * x

    val feed = Map("x" -> TensorData.fromArray(List(inp), Array(1.0, 2.0)))
    assertBlasMatchesEval(expr, feed)
  }

  test("matrix-matrix multiply: A * B") {
    val m = Dim("m", 2)
    val k = Dim("k", 3)
    val n = Dim("n", 2)
    val aData = TensorData.fromArray(List(m, k), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val bData = TensorData.fromArray(List(k, n), Array(1.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    val a = Ein.Param[Double]("A", List(m, k), aData)
    val b = Ein.Param[Double]("B", List(k, n), bData)
    val expr = a * b

    assertBlasMatchesEval(expr, Map.empty)
  }

  test("dot product: vector * vector") {
    val d = Dim("d", 3)
    val aData = TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0))
    val bData = TensorData.fromArray(List(d), Array(4.0, 5.0, 6.0))
    val a = Ein.Param[Double]("a", List(d), aData)
    val b = Ein.Param[Double]("b", List(d), bData)
    val expr = a * b

    assertBlasMatchesEval(expr, Map.empty)
  }

  // ── Element-wise operations ─────────────────────────────────────────

  test("element-wise add") {
    val d = Dim("d", 4)
    val a = Ein.Input[Double]("a", List(d))
    val b = Ein.Input[Double]("b", List(d))
    val expr = a + b
    val feed = Map(
      "a" -> TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0)),
      "b" -> TensorData.fromArray(List(d), Array(10.0, 20.0, 30.0, 40.0))
    )
    assertBlasMatchesEval(expr, feed)
  }

  test("element-wise sub") {
    val d = Dim("d", 3)
    val a = Ein.Input[Double]("a", List(d))
    val b = Ein.Input[Double]("b", List(d))
    val expr = a - b
    val feed = Map(
      "a" -> TensorData.fromArray(List(d), Array(10.0, 20.0, 30.0)),
      "b" -> TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0))
    )
    assertBlasMatchesEval(expr, feed)
  }

  test("element-wise mul") {
    val d = Dim("d", 3)
    val a = Ein.Input[Double]("a", List(d))
    val b = Ein.Input[Double]("b", List(d))
    val expr = Ein.ElemMul(a, b)
    val feed = Map(
      "a" -> TensorData.fromArray(List(d), Array(2.0, 3.0, 4.0)),
      "b" -> TensorData.fromArray(List(d), Array(5.0, 6.0, 7.0))
    )
    assertBlasMatchesEval(expr, feed)
  }

  test("element-wise add with broadcast (vector + scalar-like)") {
    val d = Dim("d", 3)
    val a = Ein.Input[Double]("a", List(d))
    val bData = TensorData.fromArray(List(Dim("s", 1)), Array(10.0))
    val b = Ein.Param[Double]("b", List(Dim("s", 1)), bData)
    val expr = a + b
    val feed = Map("a" -> TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    assertBlasMatchesEval(expr, feed)
  }

  // ── Activation tests ────────────────────────────────────────────────

  test("ReLU activation") {
    val d = Dim("d", 5)
    val x = Ein.Input[Double]("x", List(d))
    val expr = Ein.Activate(Activation.ReLU, x)
    val feed = Map("x" -> TensorData.fromArray(List(d), Array(-2.0, -1.0, 0.0, 1.0, 2.0)))
    assertBlasMatchesEval(expr, feed)
  }

  test("Sigmoid activation") {
    val d = Dim("d", 3)
    val x = Ein.Input[Double]("x", List(d))
    val expr = Ein.Activate(Activation.Sigmoid, x)
    val feed = Map("x" -> TensorData.fromArray(List(d), Array(-1.0, 0.0, 1.0)))
    assertBlasMatchesEval(expr, feed)
  }

  test("Tanh activation") {
    val d = Dim("d", 3)
    val x = Ein.Input[Double]("x", List(d))
    val expr = Ein.Activate(Activation.Tanh, x)
    val feed = Map("x" -> TensorData.fromArray(List(d), Array(-1.0, 0.0, 1.0)))
    assertBlasMatchesEval(expr, feed)
  }

  test("GELU activation") {
    val d = Dim("d", 3)
    val x = Ein.Input[Double]("x", List(d))
    val expr = Ein.Activate(Activation.GELU, x)
    val feed = Map("x" -> TensorData.fromArray(List(d), Array(-1.0, 0.0, 1.0)))
    assertBlasMatchesEval(expr, feed, tolerance = 1e-6)
  }

  test("Swish activation") {
    val d = Dim("d", 3)
    val x = Ein.Input[Double]("x", List(d))
    val expr = Ein.Activate(Activation.Swish, x)
    val feed = Map("x" -> TensorData.fromArray(List(d), Array(-1.0, 0.0, 1.0)))
    assertBlasMatchesEval(expr, feed)
  }

  // ── FillOnes / FillZeros ────────────────────────────────────────────

  test("FillOnes produces ones vector") {
    val d = Dim("d", 4)
    val expr = Ein.Ones[Double](List(d))
    assertBlasMatchesEval(expr, Map.empty)
  }

  test("FillZeros produces zeros vector") {
    val d = Dim("d", 4)
    val expr = Ein.Zeros[Double](List(d))
    assertBlasMatchesEval(expr, Map.empty)
  }

  // ── ReduceSum ───────────────────────────────────────────────────────

  test("reduce sum over one dimension") {
    val r = Dim("r", 2)
    val c = Dim("c", 3)
    val x = Ein.Input[Double]("x", List(r, c))
    val expr = Ein.ReduceSum(x, List("c"))
    val feed = Map("x" -> TensorData.fromArray(List(r, c), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    assertBlasMatchesEval(expr, feed)
  }

  // ── Broadcast ───────────────────────────────────────────────────────

  test("broadcast scalar to vector") {
    val s = Dim("s", 1)
    val d = Dim("d", 4)
    val x = Ein.Input[Double]("x", List(s))
    val expr = Ein.Broadcast(x, List(d))
    val feed = Map("x" -> TensorData.fromArray(List(s), Array(3.14)))
    assertBlasMatchesEval(expr, feed)
  }

  // ── Copy ────────────────────────────────────────────────────────────

  test("copy (transpose lowered as copy)") {
    val r = Dim("r", 2)
    val c = Dim("c", 3)
    val x = Ein.Input[Double]("x", List(r, c))
    val expr = Ein.Transpose(x, List("c", "r"))
    val feed = Map("x" -> TensorData.fromArray(List(r, c), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    // Note: Lower lowers Transpose as Copy, so this tests Copy op
    val prog = Lower.lower(expr)
    val inputData = Map("x" -> Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val result = BlasExec.run(prog, Map.empty, inputData)
    assertEquals(result.head.length, 6)
  }

  // ── MLP end-to-end ──────────────────────────────────────────────────

  test("2-layer MLP: forward pass matches EinEval") {
    val inp = Dim("inp", 2)
    val hid = Dim("hid", 3)
    val out = Dim("out", 1)

    val w1Data = TensorData.fromArray(List(hid, inp), Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    val b1Data = TensorData.fromArray(List(hid), Array(0.01, 0.02, 0.03))
    val w2Data = TensorData.fromArray(List(out, hid), Array(0.7, 0.8, 0.9))
    val b2Data = TensorData.fromArray(List(out), Array(0.04))

    val w1 = Ein.Param[Double]("W1", List(hid, inp), w1Data)
    val b1 = Ein.Param[Double]("b1", List(hid), b1Data)
    val w2 = Ein.Param[Double]("W2", List(out, hid), w2Data)
    val b2 = Ein.Param[Double]("b2", List(out), b2Data)
    val x = Ein.Input[Double]("x", List(inp))

    val h = Ein.Activate(Activation.ReLU, (w1 * x) + b1)
    val output = (w2 * h) + b2

    val feed = Map("x" -> TensorData.fromArray(List(inp), Array(1.0, 2.0)))
    assertBlasMatchesEval(output, feed)
  }

  test("block-generated network matches EinEval") {
    val inp = Dim("inp", 2)
    val input = Ein.Input[Double]("x", List(inp))
    val net = (Block.dense[Double](4, Activation.Tanh) >> Block.linear[Double](1))
      .materialize(input)

    val feed = Map("x" -> TensorData.fromArray(List(inp), Array(0.5, -0.3)))
    assertBlasMatchesEval(net, feed)
  }

  // ── Softmax ─────────────────────────────────────────────────────────

  test("softmax") {
    val d = Dim("d", 4)
    val x = Ein.Input[Double]("x", List(d))
    val expr = Ein.Softmax(x, "d")
    val feed = Map("x" -> TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0)))
    assertBlasMatchesEval(expr, feed, tolerance = 1e-7)
  }

  // ── LogSoftmax ──────────────────────────────────────────────────────

  test("logsoftmax") {
    val d = Dim("d", 4)
    val x = Ein.Input[Double]("x", List(d))
    val expr = Ein.LogSoftmax(x, "d")
    val feed = Map("x" -> TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0)))
    assertBlasMatchesEval(expr, feed, tolerance = 1e-7)
  }

  // ── Reshape ─────────────────────────────────────────────────────────

  test("reshape") {
    val d = Dim("d", 6)
    val x = Ein.Input[Double]("x", List(d))
    val target = List(Dim("r", 2), Dim("c", 3))
    val expr = Ein.Reshape(x, target)
    val feed = Map("x" -> TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    assertBlasMatchesEval(expr, feed)
  }
