package palladium.ein

class EinTraceSuite extends munit.FunSuite:

  val i = Dim("i", 3)
  val j = Dim("j", 2)

  test("single param produces 1 node") {
    val w = Ein.Param("W", List(i, j), TensorData.zeros[Double](List(i, j)))
    val graph = EinTrace.forward(w)
    assertEquals(graph.nodes.size, 1)
    assertEquals(graph.nodes(0).kind, EinNodeKind.Parameter("W"))
    assertEquals(graph.nodes(0).outputDims, List(i, j))
  }

  test("single input produces 1 node") {
    val x = Ein.Input[Double]("x", List(i))
    val graph = EinTrace.forward(x)
    assertEquals(graph.nodes.size, 1)
    assertEquals(graph.nodes(0).kind, EinNodeKind.Input("x"))
    assertEquals(graph.nodes(0).outputDims, List(i))
  }

  test("contract produces 3 nodes (left, right, contract)") {
    val w = Ein.Param("W", List(i, j), TensorData.zeros[Double](List(i, j)))
    val x = Ein.Param("x", List(j), TensorData.zeros[Double](List(j)))
    val graph = EinTrace.forward(w * x)
    assertEquals(graph.nodes.size, 3)
    assertEquals(graph.nodes(2).kind, EinNodeKind.Contract)
    assertEquals(graph.nodes(2).outputDims, List(i))
  }

  test("dense layer relu(W*x + b) produces 6 nodes") {
    val w = Ein.Param("W", List(i, j), TensorData.zeros[Double](List(i, j)))
    val x = Ein.Param("x", List(j), TensorData.zeros[Double](List(j)))
    val b = Ein.Param("b", List(i), TensorData.zeros[Double](List(i)))
    val expr = Ein.Activate(Activation.ReLU, (w * x) + b)
    val graph = EinTrace.forward(expr)
    // W, x, contract, b, +, ReLU = 6 nodes
    assertEquals(graph.nodes.size, 6)
    assertEquals(graph.nodes(graph.rootId).kind, EinNodeKind.Activate(Activation.ReLU))
  }

  test("param deduplication: w * w produces 2 nodes not 3") {
    val w = Ein.Param("W", List(i), TensorData.zeros[Double](List(i)))
    val graph = EinTrace.forward(w * w)
    // W (deduplicated), contract = 2 nodes
    assertEquals(graph.nodes.size, 2)
    val contract = graph.nodes(1)
    assertEquals(contract.inputs, List(0, 0))
  }

  test("input deduplication works") {
    val x = Ein.Input[Double]("x", List(i))
    val graph = EinTrace.forward(x + x)
    // x (deduplicated), + = 2 nodes
    assertEquals(graph.nodes.size, 2)
    val add = graph.nodes(1)
    assertEquals(add.inputs, List(0, 0))
  }

  test("output dims flow through multi-layer network") {
    val inp = Dim("inp", 2)
    val hid = Dim("hid", 3)
    val out = Dim("out", 1)
    val w1 = Ein.Param("W1", List(hid, inp), TensorData.zeros[Double](List(hid, inp)))
    val b1 = Ein.Param("b1", List(hid), TensorData.zeros[Double](List(hid)))
    val w2 = Ein.Param("W2", List(out, hid), TensorData.zeros[Double](List(out, hid)))
    val b2 = Ein.Param("b2", List(out), TensorData.zeros[Double](List(out)))
    val x = Ein.Param("x", List(inp), TensorData.zeros[Double](List(inp)))

    val h = Ein.Activate(Activation.ReLU, (w1 * x) + b1)
    val y = (w2 * h) + b2
    val graph = EinTrace.forward(y)

    // Root should have output dims [out]
    val root = graph.nodes(graph.rootId)
    assertEquals(root.outputDims, List(out))

    // Find the ReLU node — should have dims [hid]
    val reluNode = graph.nodes.find(_.kind == EinNodeKind.Activate(Activation.ReLU)).get
    assertEquals(reluNode.outputDims, List(hid))
  }

  test("fill node traced correctly") {
    val d = Dim("d", 3)
    val fill = Ein.Fill(1.0, List(d))
    val graph = EinTrace.forward(fill)
    assertEquals(graph.nodes.size, 1)
    assertEquals(graph.nodes(0).kind, EinNodeKind.Fill)
    assertEquals(graph.nodes(0).outputDims, List(d))
  }

  test("reduce sum traced correctly") {
    val di = Dim("i", 2)
    val dj = Dim("j", 3)
    val m = Ein.Param("M", List(di, dj), TensorData.zeros[Double](List(di, dj)))
    val graph = EinTrace.forward(Ein.ReduceSum(m, List("j")))
    assertEquals(graph.nodes.size, 2)
    assertEquals(graph.nodes(1).kind, EinNodeKind.ReduceSum(List("j")))
    assertEquals(graph.nodes(1).outputDims, List(di))
  }

  test("syntax extension works") {
    import EinTrace.syntax.*
    val w = Ein.Param("W", List(i, j), TensorData.zeros[Double](List(i, j)))
    val graph = w.toGraph
    assertEquals(graph.nodes.size, 1)
  }
