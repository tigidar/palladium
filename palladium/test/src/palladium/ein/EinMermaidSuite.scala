package palladium.ein

class EinMermaidSuite extends munit.FunSuite:

  test("ops output starts with flowchart TD") {
    val d = Dim("d", 3)
    val w = Ein.Param("W", List(d), TensorData.zeros[Double](List(d)))
    val graph = EinTrace.forward(w)
    val mermaid = EinMermaid.renderOps(graph)
    assert(mermaid.startsWith("flowchart TD"), s"Expected flowchart TD, got: ${mermaid.take(50)}")
  }

  test("ops includes shape annotations") {
    val i = Dim("i", 3)
    val j = Dim("j", 2)
    val w = Ein.Param("W", List(i, j), TensorData.zeros[Double](List(i, j)))
    val x = Ein.Param("x", List(j), TensorData.zeros[Double](List(j)))
    val graph = EinTrace.forward(w * x)
    val mermaid = EinMermaid.renderOps(graph)
    assert(mermaid.contains("[3 x 2]"), s"Expected shape [3 x 2] in ops output")
    assert(mermaid.contains("[3]"), s"Expected shape [3] in ops output")
    assert(mermaid.contains("contract"), s"Expected contract node in ops output")
  }

  test("ops includes activation labels") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.zeros[Double](List(d)))
    val graph = EinTrace.forward(Ein.Activate(Activation.ReLU, v))
    val mermaid = EinMermaid.renderOps(graph)
    assert(mermaid.contains("ReLU"), s"Expected ReLU label in ops output")
  }

  test("arch output starts with flowchart TD") {
    val features = Dim("j", 3)
    val hidden = Dim("i", 2)
    val w = Ein.Param("W", List(hidden, features), TensorData.zeros[Double](List(hidden, features)))
    val b = Ein.Param("b", List(hidden), TensorData.zeros[Double](List(hidden)))
    val x = Ein.Param("x", List(features), TensorData.zeros[Double](List(features)))
    val expr = Ein.Activate(Activation.ReLU, (w * x) + b)

    val graph = EinTrace.forward(expr)
    val arch = EinArch.detect(graph)
    val mermaid = EinMermaid.renderArch(arch)
    assert(mermaid.startsWith("flowchart TD"), s"Expected flowchart TD, got: ${mermaid.take(50)}")
  }

  test("arch includes layer labels and param info") {
    val features = Dim("j", 3)
    val hidden = Dim("i", 2)
    val w = Ein.Param("W", List(hidden, features), TensorData.zeros[Double](List(hidden, features)))
    val b = Ein.Param("b", List(hidden), TensorData.zeros[Double](List(hidden)))
    val x = Ein.Param("x", List(features), TensorData.zeros[Double](List(features)))
    val expr = Ein.Activate(Activation.ReLU, (w * x) + b)

    val graph = EinTrace.forward(expr)
    val arch = EinArch.detect(graph)
    val mermaid = EinMermaid.renderArch(arch)
    assert(mermaid.contains("Dense + ReLU"), s"Expected 'Dense + ReLU' label")
    assert(mermaid.contains("params: 8"), s"Expected 'params: 8' (6 + 2)")
    assert(mermaid.contains("W:"), s"Expected weight param info")
    assert(mermaid.contains("b:"), s"Expected bias param info")
  }

  test("arch connections show shape transformations for 2-layer network") {
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
    val arch = EinArch.detect(graph)
    val mermaid = EinMermaid.renderArch(arch)

    assert(arch.layers.size == 2, s"Expected 2 layers, got ${arch.layers.size}")
    assert(mermaid.contains("layer0"), "Expected layer0 node")
    assert(mermaid.contains("layer1"), "Expected layer1 node")
    assert(mermaid.contains("-->"), "Expected connections")
  }

  test("syntax extensions work") {
    import EinMermaid.syntax.*
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.zeros[Double](List(d)))
    val graph = EinTrace.forward(v)
    val opsMermaid = graph.toOpsMermaid
    assert(opsMermaid.startsWith("flowchart TD"))

    val arch = EinArch.detect(graph)
    val archMermaid = arch.toArchMermaid
    assert(archMermaid.startsWith("flowchart TD"))
  }
