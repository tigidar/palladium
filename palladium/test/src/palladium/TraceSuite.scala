package palladium

class TraceSuite extends munit.FunSuite:

  val tolerance = 1e-5

  // ============================================
  // Forward tracing tests
  // ============================================

  test("forward: single variable") {
    val x = Value.variable("x", 2.0)
    val graph = Trace.forward(x)
    assertEquals(graph.nodes.size, 1)
    assertEquals(graph.nodes(graph.rootId).value, 2.0)
    assertEquals(graph.nodes(graph.rootId).kind, NodeKind.Variable("x"))
  }

  test("forward: x + y evaluates correctly at every node") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val graph = Trace.forward(x + y)
    assertEquals(graph.nodes.size, 3)
    assertEquals(graph.nodes(graph.rootId).value, 5.0)
  }

  test("forward: x * y + x deduplicates x") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val graph = Trace.forward(x * y + x)
    // x, y, *, + → 4 nodes (x appears once)
    assertEquals(graph.nodes.size, 4)
    assertEquals(graph.nodes(graph.rootId).value, 8.0)
  }

  test("forward: x * x deduplicates x") {
    val x = Value.variable("x", 3.0)
    val graph = Trace.forward(x * x)
    // x, * → 2 nodes
    assertEquals(graph.nodes.size, 2)
    assertEquals(graph.nodes(graph.rootId).value, 9.0)
  }

  test("forward: nested expression 3x^2 - 4x + 5") {
    val x = Value.variable("x", 2.0)
    val expr =
      Value.const(3) * (x ~^ Value.const(2)) - Value.const(4) * x + Value
        .const(5)
    val graph = Trace.forward(expr)
    // 3*x^2 - 4*x + 5 = 12 - 8 + 5 = 9
    assertEquals(graph.nodes(graph.rootId).value, 9.0)
  }

  // ============================================
  // Backward over traced DAG — must match Grad.backward
  // ============================================

  test("backward: x * y matches Grad.backward") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val expr = x * y
    val graph = Trace.forward(expr)
    val grads = Trace.backward(graph)
    val varGrads = Trace.varGrads(graph, grads)
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
    assertEqualsDouble(varGrads("y"), expected("y"), tolerance)
  }

  test("backward: x * x + x matches Grad.backward") {
    val x = Value.variable("x", 2.0)
    val expr = x * x + x
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    // d/dx (x^2 + x) = 2x + 1 = 5
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: division x / y matches Grad.backward") {
    val x = Value.variable("x", 6.0)
    val y = Value.variable("y", 2.0)
    val expr = x / y
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
    assertEqualsDouble(varGrads("y"), expected("y"), tolerance)
  }

  test("backward: fast division x /~ y matches Grad.backward") {
    val x = Value.variable("x", 6.0)
    val y = Value.variable("y", 2.0)
    val expr = x /~ y
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
    assertEqualsDouble(varGrads("y"), expected("y"), tolerance)
  }

  test("backward: power x^y matches Grad.backward") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val expr = x ~^ y
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
    assertEqualsDouble(varGrads("y"), expected("y"), tolerance)
  }

  test("backward: negation -x matches Grad.backward") {
    val x = Value.variable("x", 5.0)
    val expr = -x
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: log(x) matches Grad.backward") {
    val x = Value.variable("x", 2.0)
    val expr = x.log
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: exp(x) matches Grad.backward") {
    val x = Value.variable("x", 2.0)
    val expr = x.exp
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: tanh(x) matches Grad.backward") {
    val x = Value.variable("x", 1.0)
    val expr = x.tanh
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: sigmoid(x) matches Grad.backward") {
    val x = Value.variable("x", 1.0)
    val expr = x.sigmoid
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: relu(x) matches Grad.backward for x > 0") {
    val x = Value.variable("x", 3.0)
    val expr = x.relu
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: relu(x) matches Grad.backward for x < 0") {
    val x = Value.variable("x", -2.0)
    val expr = x.relu
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: sin(x) matches Grad.backward") {
    val x = Value.variable("x", 1.0)
    val expr = x.sin
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: cos(x) matches Grad.backward") {
    val x = Value.variable("x", 1.0)
    val expr = x.cos
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: abs(x) matches Grad.backward for x > 0") {
    val x = Value.variable("x", 3.0)
    val expr = x.abs
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: abs(x) matches Grad.backward for x < 0") {
    val x = Value.variable("x", -3.0)
    val expr = x.abs
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: polynomial 3x^2 - 4x + 5 matches Grad.backward") {
    val x = Value.variable("x", 2.0)
    val expr =
      Value.const(3) * (x ~^ Value.const(2)) - Value.const(4) * x + Value
        .const(5)
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
  }

  test("backward: multi-variable x*y + x*z matches Grad.backward") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val z = Value.variable("z", 4.0)
    val expr = x * y + x * z
    val graph = Trace.forward(expr)
    val varGrads = Trace.varGrads(graph, Trace.backward(graph))
    val expected = Grad.backward(expr)
    assertEqualsDouble(varGrads("x"), expected("x"), tolerance)
    assertEqualsDouble(varGrads("y"), expected("y"), tolerance)
    assertEqualsDouble(varGrads("z"), expected("z"), tolerance)
  }

  // ============================================
  // Intermediate gradients (the value-add over Grad.backward)
  // ============================================

  test("backward: every node gets a gradient") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val expr = x * y + x
    val graph = Trace.forward(expr)
    val grads = Trace.backward(graph)
    // Every node should have an entry
    for node <- graph.nodes do
      assert(grads.contains(node.id), s"missing gradient for node ${node.id}")
  }

  test("backward: root gradient is 1.0") {
    val x = Value.variable("x", 2.0)
    val expr = x * x
    val graph = Trace.forward(expr)
    val grads = Trace.backward(graph)
    assertEquals(grads(graph.rootId), 1.0)
  }

  // ============================================
  // Syntax extension tests
  // ============================================

  test("syntax: toGraph and varGradients") {
    import Trace.syntax.*
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val expr = x * y
    val graph = expr.toGraph
    val grads = graph.varGradients
    assertEqualsDouble(grads("x"), 3.0, tolerance)
    assertEqualsDouble(grads("y"), 2.0, tolerance)
  }

  // ============================================
  // Mermaid rendering tests
  // ============================================

  test("mermaid: renders valid flowchart header") {
    val x = Value.variable("x", 2.0)
    val graph = Trace.forward(x * Value.variable("y", 3.0))
    val output = Mermaid.render(graph)
    assert(output.startsWith("flowchart TD\n"))
  }

  test("mermaid: includes all nodes and edges") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val graph = Trace.forward(x + y)
    val output = Mermaid.render(graph)
    // 3 nodes: x, y, +
    assert(output.contains("n0"))
    assert(output.contains("n1"))
    assert(output.contains("n2"))
    // 2 edges
    assert(output.contains("n0 --> n2"))
    assert(output.contains("n1 --> n2"))
  }

  test("mermaid: renders with gradients") {
    val x = Value.variable("x", 2.0)
    val y = Value.variable("y", 3.0)
    val expr = x * y
    val graph = Trace.forward(expr)
    val grads = Trace.backward(graph)
    val output = Mermaid.render(graph, grads)
    // We are not able to match against "∇", but I think this is \nabla
    assert(output.contains("\\nabla"))
  }

  test("mermaid: syntax extension") {
    import Mermaid.syntax.*
    val x = Value.variable("x", 2.0)
    val graph = Trace.forward(x + Value(1.0))
    val output = graph.toMermaid
    assert(output.startsWith("flowchart TD\n"))
  }
