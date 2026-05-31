package palladium.ein

import palladium.ein.EinDsl.*

class EinDslSuite extends munit.FunSuite:

  val tolerance = 1e-9

  // --- Int.dim ---

  test("Int.dim captures name and size") {
    val features = 3.dim
    assertEquals(features, Dim("features", 3))
  }

  test("Int.dim captures different names") {
    val inp = 2.dim
    val out = 1.dim
    val hidden = 128.dim
    assertEquals(inp, Dim("inp", 2))
    assertEquals(out, Dim("out", 1))
    assertEquals(hidden, Dim("hidden", 128))
  }

  // --- input ---

  test("input captures variable name") {
    val features = 3.dim
    val X = input[Double](features)
    X match
      case Ein.Input(id, dims) =>
        assertEquals(id, "X")
        assertEquals(dims, List(features))
      case _ => fail("Expected Ein.Input")
  }

  // --- weight ---

  test("weight creates param with from → to dims") {
    val inp = 2.dim
    val out = 3.dim
    val w = weight(inp → out, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    w match
      case Ein.Param(id, dims, td) =>
        assertEquals(id, "w")
        assertEquals(dims, List(out, inp))  // stored as (to, from) for row-major convention
        assertEquals(td.data.toList, List(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
      case _ => fail("Expected Ein.Param")
  }

  test("weight with single output") {
    val inp = 2.dim
    val out = 1.dim
    val w = weight(inp → out, -3.0, 1.0)
    w match
      case Ein.Param(id, dims, td) =>
        assertEquals(id, "w")
        assertEquals(dims, List(out, inp))
        assertEquals(td.data.toList, List(-3.0, 1.0))
      case _ => fail("Expected Ein.Param")
  }

  // --- bias ---

  test("bias creates param with single dim") {
    val out = 3.dim
    val b = bias(out, 0.1, 0.2, 0.3)
    b match
      case Ein.Param(id, dims, td) =>
        assertEquals(id, "b")
        assertEquals(dims, List(out))
        assertEquals(td.data.toList, List(0.1, 0.2, 0.3))
      case _ => fail("Expected Ein.Param")
  }

  test("bias with single output") {
    val out = 1.dim
    val b = bias(out, 6.88)
    b match
      case Ein.Param(id, dims, td) =>
        assertEquals(id, "b")
        assertEquals(dims, List(out))
        assertEqualsDouble(td.data(0), 6.88, tolerance)
      case _ => fail("Expected Ein.Param")
  }

  // --- activations ---

  test("relu/sigmoid/tanh create Activate nodes") {
    val d = 2.dim
    val x = input[Double](d)
    assert(relu(x).isInstanceOf[Ein.Activate[Double]])
    assert(sigmoid(x).isInstanceOf[Ein.Activate[Double]])
    assert(tanh(x).isInstanceOf[Ein.Activate[Double]])
  }

  // --- operators ---

  test("operator precedence: W * X + B parses as (W * X) + B") {
    val features = 3.dim
    val hidden = Dim("hidden", 2)
    val x = input[Double](features)
    val W = weight(features → hidden, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    val B = bias(hidden, 0.1, 0.2)
    val expr = W * x + B
    expr match
      case Ein.ElemAdd(Ein.Contract(_, _), _) => () // correct
      case _ => fail(s"Expected ElemAdd(Contract(...), ...), got: $expr")
  }

  test("× operator works same as *") {
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val b = Ein.Param("b", List(d), TensorData.fromArray(List(d), Array(4.0, 5.0, 6.0)))
    val dot1 = a × b
    val dot2 = a * b
    val r1 = EinEval.eval(dot1)
    val r2 = EinEval.eval(dot2)
    assertEqualsDouble(r1.data(0), r2.data(0), tolerance)
  }

  // --- extensions ---

  test("sumOver creates ReduceSum") {
    val d = 3.dim
    val x = input[Double](d)
    val expr = x.sumOver("d")
    expr match
      case Ein.ReduceSum(_, List("d")) => ()
      case _ => fail("Expected ReduceSum")
  }

  test("syntax.eval works") {
    import EinDsl.syntax.*
    val d = 2.dim
    val v = weight(d → d, 3.0, 4.0, 5.0, 6.0)
    val result = v.eval()
    assertEqualsDouble(result.data(0), 3.0, tolerance)
    assertEqualsDouble(result.data(1), 4.0, tolerance)
  }

  test("syntax.backward works") {
    import EinDsl.syntax.*
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val b = Ein.Param("b", List(d), TensorData.fromArray(List(d), Array(4.0, 5.0, 6.0)))
    val dot = a × b
    val grads = dot.backward()
    assertEqualsDouble(grads("a").data(0), 4.0, 1e-5)
  }

  // --- weight/bias varargs ---

  test("weight with varargs") {
    val inp = 2.dim
    val out = 3.dim
    val w = weight(inp → out, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    w match
      case Ein.Param(id, dims, data) =>
        assertEquals(id, "w")
        assertEquals(dims, List(out, inp))
        assertEquals(data.data.toList, List(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
      case _ => fail("Expected Ein.Param")
  }

  test("bias with varargs") {
    val out = 3.dim
    val b = bias(out, 0.1, 0.2, 0.3)
    b match
      case Ein.Param(id, dims, data) =>
        assertEquals(id, "b")
        assertEquals(dims, List(out))
        assertEquals(data.data.toList, List(0.1, 0.2, 0.3))
      case _ => fail("Expected Ein.Param")
  }

  // --- input.data feed factory ---

  test("input.data returns feed map") {
    val inp = 2.dim
    val x = input[Double](inp)
    val feed = x.data(2.0, 0.0)
    assertEquals(feed.size, 1)
    assertEquals(feed("x").dims, List(inp))
    assertEquals(feed("x").data.toList, List(2.0, 0.0))
  }

  test("eval accepts multiple .data feeds with comma") {
    import EinDsl.syntax.*

    val inp = 2.dim
    val out = 1.dim

    val x = input[Double](inp)
    val target = input[Double](out)

    // x - target (just a simple expression with two inputs)
    val diff = x - target.broadcast(inp)
    val result = diff.eval(x.data(3.0, 1.0), target.data(2.0))

    // [3 - 2, 1 - 2] = [1, -1]
    assertEqualsDouble(result.data(0), 1.0, tolerance)
    assertEqualsDouble(result.data(1), -1.0, tolerance)
  }

  // --- end-to-end ---

  test("full neuron with new DSL: weight * input + bias") {
    import EinDsl.syntax.*

    val inp = 2.dim
    val out = 1.dim

    val x = input[Double](inp)
    val w = weight(inp → out, -3.0, 1.0)
    val b = bias(out, 6.8813735870195432)

    val o = tanh(w * x + b)

    val result = o.eval(x.data(2.0, 0.0))

    // tanh((-3.0 * 2.0) + (1.0 * 0.0) + 6.8813735870195432) = tanh(0.8813735870195432)
    assertEqualsDouble(result.data(0), Math.tanh(0.8813735870195432), tolerance)
  }

  test("multi-output neuron with new DSL") {
    import EinDsl.syntax.*

    val inp = 2.dim
    val out = 3.dim

    val x = input[Double](inp)
    val w = weight(inp → out, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    val b = bias(out, 0.0, 0.0, 0.0)

    val y = relu(w * x + b)

    val result = y.eval(x.data(3.0, -2.0))

    // out0: relu(1*3 + 0*(-2) + 0) = relu(3) = 3
    // out1: relu(0*3 + 1*(-2) + 0) = relu(-2) = 0
    // out2: relu(1*3 + 1*(-2) + 0) = relu(1) = 1
    assertEqualsDouble(result.data(0), 3.0, tolerance)
    assertEqualsDouble(result.data(1), 0.0, tolerance)
    assertEqualsDouble(result.data(2), 1.0, tolerance)
  }

  // --- Init: weight/bias with initializers ---

  test("weight with uniform initializer") {
    val inp = 3.dim
    val out = 2.dim
    given rng: java.util.Random = java.util.Random(42)
    val w = weight(inp → out, uniform(-1, 1))
    w match
      case Ein.Param(_, dims, td) =>
        assertEquals(dims, List(out, inp))
        assertEquals(td.data.length, 6)
        // all values in [-1, 1]
        td.data.foreach(v => assert(v >= -1.0 && v <= 1.0, s"$v out of range"))
      case _ => fail("Expected Ein.Param")
  }

  test("weight with uniform is reproducible with same seed") {
    val inp = 3.dim
    val out = 2.dim
    val d1 = {
      given rng: java.util.Random = java.util.Random(42)
      weight(inp → out, uniform(-1, 1))
    }
    val d2 = {
      given rng: java.util.Random = java.util.Random(42)
      weight(inp → out, uniform(-1, 1))
    }
    (d1, d2) match
      case (Ein.Param(_, _, td1), Ein.Param(_, _, td2)) =>
        assertEquals(td1.data.toList, td2.data.toList)
      case _ => fail("Expected Ein.Param")
  }

  test("weight with xavier initializer") {
    val inp = 100.dim
    val out = 50.dim
    given rng: java.util.Random = java.util.Random(42)
    val w = weight(inp → out, xavier)
    w match
      case Ein.Param(_, _, td) =>
        // Xavier range: sqrt(6 / (fan_in + fan_out)) = sqrt(6/150) ≈ 0.2
        val limit = Math.sqrt(6.0 / (100 + 50))
        td.data.foreach(v => assert(v >= -limit && v <= limit, s"$v out of xavier range"))
      case _ => fail("Expected Ein.Param")
  }

  test("bias with zeros initializer") {
    val out = 3.dim
    val b = bias(out, zeros)
    b match
      case Ein.Param(_, _, td) =>
        assertEquals(td.data.toList, List(0.0, 0.0, 0.0))
      case _ => fail("Expected Ein.Param")
  }

  test("weight with zeros initializer") {
    val inp = 2.dim
    val out = 3.dim
    val w = weight(inp → out, zeros)
    w match
      case Ein.Param(_, _, td) =>
        assertEquals(td.data.toList, List(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
      case _ => fail("Expected Ein.Param")
  }

  // --- .square, .sum, ** ---

  test(".square is element-wise squaring") {
    import EinDsl.syntax.*
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(2.0, -3.0, 4.0)))
    val result = v.square.eval()
    assertEquals(result.data.toList, List(4.0, 9.0, 16.0))
  }

  test(".sum reduces all dims to scalar") {
    import EinDsl.syntax.*
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val result = v.sum.eval()
    assertEqualsDouble(result.data(0), 6.0, tolerance)
    assertEquals(result.dims, Nil)
  }

  test("** 2 is same as .square") {
    import EinDsl.syntax.*
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(2.0, -3.0, 4.0)))
    val r1 = v.square.eval()
    val r2 = (v ** 2).eval()
    assertEquals(r1.data.toList, r2.data.toList)
  }

  test("** 3 is element-wise cube") {
    import EinDsl.syntax.*
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(2.0, -3.0, 4.0)))
    val result = (v ** 3).eval()
    assertEquals(result.data.toList, List(8.0, -27.0, 64.0))
  }

  test("(diff).square.sum computes MSE-like loss") {
    import EinDsl.syntax.*
    val d = Dim("d", 2)
    val pred = Ein.Param("pred", List(d), TensorData.fromArray(List(d), Array(0.5, -0.5)))
    val target = Ein.Param("target", List(d), TensorData.fromArray(List(d), Array(1.0, -1.0)))
    val loss = (pred - target).square.sum
    val result = loss.eval()
    // (0.5-1)² + (-0.5-(-1))² = 0.25 + 0.25 = 0.5
    assertEqualsDouble(result.data(0), 0.5, tolerance)
  }

  // --- .parameters ---

  test(".parameters collects all params from expression tree") {
    val inp = 2.dim
    val out = 1.dim
    val x = input[Double](inp)
    val w = weight(inp → out, -3.0, 1.0)
    val b = bias(out, 6.88)

    val expr = tanh(w * x + b)
    val params = expr.parameters

    assertEquals(params.size, 2)
    assert(params.contains("w"))
    assert(params.contains("b"))
    assertEquals(params("w").data.toList, List(-3.0, 1.0))
  }

  test(".parameters deduplicates shared params") {
    val d = Dim("d", 2)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0)))
    // v appears twice in the tree
    val expr = v + v
    val params = expr.parameters
    assertEquals(params.size, 1)
  }

  test(".parameters on Block DSL network") {
    import Block.*
    given rng: java.util.Random = java.util.Random(42)

    val inp = 3.dim
    val x = input[Double](inp)
    val net = x >> dense(4, Activation.Tanh, uniform(-1, 1)) >> dense(1, Activation.Tanh, uniform(-1, 1))
    val expr = net.materialize

    val params = expr.parameters
    // 2 layers × (1 weight + 1 bias) = 4 params
    assertEquals(params.size, 4)
    assert(params.contains("block0/W"))
    assert(params.contains("block0/b"))
    assert(params.contains("block1/W"))
    assert(params.contains("block1/b"))
  }
