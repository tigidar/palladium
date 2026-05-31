package palladium.ein

class EinArchSuite extends munit.FunSuite:

  test("single dense + relu detected as 1 Dense layer with activation") {
    val features = Dim("j", 3)
    val hidden = Dim("i", 2)
    val w = Ein.Param("W", List(hidden, features), TensorData.zeros[Double](List(hidden, features)))
    val b = Ein.Param("b", List(hidden), TensorData.zeros[Double](List(hidden)))
    val x = Ein.Param("x", List(features), TensorData.zeros[Double](List(features)))
    val expr = Ein.Activate(Activation.ReLU, (w * x) + b)

    val graph = EinTrace.forward(expr)
    val arch = EinArch.detect(graph)

    assertEquals(arch.layers.size, 1)
    arch.layers(0).layerType match
      case LayerType.Dense(inDim, outDim, Some(Activation.ReLU)) =>
        assertEquals(inDim, features)
        assertEquals(outDim, hidden)
      case other => fail(s"Expected Dense + ReLU, got $other")
    assertEquals(arch.layers(0).params.size, 2)
  }

  test("2-layer network detected as 2 layers with correct param counts") {
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

    assertEquals(arch.layers.size, 2)

    // Layer 0: Dense + ReLU (W1: 3x2=6, b1: 3)
    arch.layers(0).layerType match
      case LayerType.Dense(_, _, Some(Activation.ReLU)) => ()
      case other => fail(s"Expected Dense + ReLU for layer 0, got $other")
    val layer0Params = arch.layers(0).params.map(_.elementCount).sum
    assertEquals(layer0Params, 9) // 6 + 3

    // Layer 1: Dense linear (W2: 1x3=3, b2: 1)
    arch.layers(1).layerType match
      case LayerType.Dense(_, _, None) => ()
      case other => fail(s"Expected Dense (linear) for layer 1, got $other")
    val layer1Params = arch.layers(1).params.map(_.elementCount).sum
    assertEquals(layer1Params, 4) // 3 + 1

    assertEquals(arch.totalParams, 13)
  }

  test("dense without activation detected as linear output layer") {
    val features = Dim("j", 3)
    val output = Dim("i", 1)
    val w = Ein.Param("W", List(output, features), TensorData.zeros[Double](List(output, features)))
    val b = Ein.Param("b", List(output), TensorData.zeros[Double](List(output)))
    val x = Ein.Param("x", List(features), TensorData.zeros[Double](List(features)))
    val expr = (w * x) + b

    val graph = EinTrace.forward(expr)
    val arch = EinArch.detect(graph)

    assertEquals(arch.layers.size, 1)
    arch.layers(0).layerType match
      case LayerType.Dense(_, _, None) => ()
      case other => fail(s"Expected Dense (linear), got $other")
  }

  test("total param count is sum of all weight and bias element counts") {
    val inp = Dim("inp", 4)
    val hid = Dim("hid", 8)
    val out = Dim("out", 2)

    val w1 = Ein.Param("W1", List(hid, inp), TensorData.zeros[Double](List(hid, inp)))
    val b1 = Ein.Param("b1", List(hid), TensorData.zeros[Double](List(hid)))
    val w2 = Ein.Param("W2", List(out, hid), TensorData.zeros[Double](List(out, hid)))
    val b2 = Ein.Param("b2", List(out), TensorData.zeros[Double](List(out)))
    val x = Ein.Param("x", List(inp), TensorData.zeros[Double](List(inp)))

    val h = Ein.Activate(Activation.Sigmoid, (w1 * x) + b1)
    val y = (w2 * h) + b2

    val graph = EinTrace.forward(y)
    val arch = EinArch.detect(graph)

    // W1: 8*4=32, b1: 8, W2: 2*8=16, b2: 2 = 58
    assertEquals(arch.totalParams, 58)
  }

  test("connections between layers are detected") {
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

    assert(arch.connections.nonEmpty, "Should have connections between layers")
    assert(arch.connections.contains((0, 1)), s"Expected connection (0,1), got ${arch.connections}")
  }
