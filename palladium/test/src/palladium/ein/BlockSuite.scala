package palladium.ein

class BlockSuite extends munit.FunSuite:

  // -- helpers --
  private val inp2 = Dim("inp", 2)
  private def input2 = Ein.Input[Double]("x", List(inp2))

  private def feedInput(values: Double*): Map[String, TensorData[Double]] =
    Map("x" -> TensorData.fromArray(List(inp2), values.toArray))

  // -- 1. Single dense block materializes to correct Ein tree --

  test("single dense block produces Activate(Contract + bias)") {
    val block = Block.dense[Double](3, Activation.ReLU)
    val net = block.materialize(input2)

    net match
      case Ein.Activate(Activation.ReLU, Ein.ElemAdd(Ein.Contract(w, Ein.Input("x", _)), b)) =>
        w match
          case Ein.Param(id, dims, _) =>
            assert(id == "block0/W")
            assertEquals(dims.length, 2)
            assertEquals(dims(0), Dim("block0_out", 3))
            assertEquals(dims(1), inp2)
          case other => fail(s"Expected Param for weight, got $other")
        b match
          case Ein.Param(id, dims, _) =>
            assert(id == "block0/b")
            assertEquals(dims, List(Dim("block0_out", 3)))
          case other => fail(s"Expected Param for bias, got $other")
      case other => fail(s"Expected Activate(ReLU, Add(Contract, bias)), got $other")
  }

  // -- 2. >> sequence chains blocks correctly --

  test(">> sequence chains two blocks") {
    val b1 = Block.dense[Double](4, Activation.ReLU)
    val b2 = Block.linear[Double](1)
    val net = (b1 >> b2).materialize(input2)

    // Output should be a linear layer (ElemAdd of Contract + bias)
    net match
      case Ein.ElemAdd(Ein.Contract(w2, inner), b2Node) =>
        w2 match
          case Ein.Param("block1/W", _, _) => ()
          case other => fail(s"Expected block1/W, got $other")
        b2Node match
          case Ein.Param("block1/b", _, _) => ()
          case other => fail(s"Expected block1/b, got $other")
        // inner should be the ReLU layer output
        inner match
          case Ein.Activate(Activation.ReLU, _) => ()
          case other => fail(s"Expected ReLU activation in inner, got $other")
      case other => fail(s"Expected linear output layer, got $other")
  }

  // -- 3. * repeat produces N blocks with unique scoped params --

  test("* repeat produces N blocks with unique scoped params") {
    val block = Block.dense[Double](4, Activation.ReLU)
    val net = (block * 3).materialize(input2)

    // Collect all param names
    val params = collectParams(net)
    val paramNames = params.map(_._1).toSet

    assertEquals(paramNames, Set(
      "block0/W", "block0/b",
      "block1/W", "block1/b",
      "block2/W", "block2/b"
    ))
  }

  // -- 4. .residual produces input + block(input) --

  test("residual produces ElemAdd(input, block(input))") {
    val block = Block.dense[Double](2, Activation.ReLU).residual
    val net = block.materialize(input2)

    net match
      case Ein.ElemAdd(inputSide, blockSide) =>
        // inputSide should be the original Input
        inputSide match
          case Ein.Input("x", _) => ()
          case other => fail(s"Expected Input on left of residual add, got $other")
        // blockSide should be Activate(ReLU, ...)
        blockSide match
          case Ein.Activate(Activation.ReLU, _) => ()
          case other => fail(s"Expected ReLU on right of residual add, got $other")
        // The block's output dim should match input's dim name for correct element-wise add
        assertEquals(inputSide.outputDims.map(_.name), blockSide.outputDims.map(_.name))
      case other => fail(s"Expected ElemAdd for residual, got $other")
  }

  // -- 5. Eval through materialized block gives correct numerical output --

  test("eval through fn block produces correct output") {
    val doubler = Block.fn[Double](x => x + x)
    val net = doubler.materialize(input2)
    val result = EinEval.eval(net, feedInput(1.0, 2.0))

    assertEquals(result.dims, List(inp2))
    assertArrayEquals(result.data, Array(2.0, 4.0))
  }

  test("eval through dense block with zero weights gives activation(0)") {
    // With zero weights: output = sigmoid(0*x + 0) = sigmoid(0) = 0.5
    val block = Block.dense[Double](3, Activation.Sigmoid)
    val net = block.materialize(input2)
    val result = EinEval.eval(net, feedInput(1.0, 2.0))

    assertEquals(result.dims.length, 1)
    assertEquals(result.dims.head.size, 3)
    result.data.foreach { v =>
      assertEqualsDouble(v, 0.5, 1e-10)
    }
  }

  // -- 6. Backward pass computes gradients for all scoped params --

  test("backward pass computes gradients for block params") {
    val block = Block.dense[Double](3, Activation.Sigmoid)
    val net = block.materialize(input2)
    val grads = EinGrad.backward(net, feedInput(1.0, 2.0))

    // Should have gradients for block0/W and block0/b
    assert(grads.contains("block0/W"), s"Missing grad for block0/W, got keys: ${grads.keys}")
    assert(grads.contains("block0/b"), s"Missing grad for block0/b, got keys: ${grads.keys}")

    // Gradients should be non-zero (sigmoid'(0) = 0.25, input is non-zero)
    assert(grads("block0/W").data.exists(_ != 0.0), "W gradient should be non-zero")
    assert(grads("block0/b").data.exists(_ != 0.0), "b gradient should be non-zero")
  }

  // -- 7. EinArch detects layers in block-generated networks --

  test("EinArch detects dense layers in block-generated network") {
    val net = (Block.dense[Double](4, Activation.ReLU) >> Block.linear[Double](1))
      .materialize(input2)

    val graph = EinTrace.forward(net)
    val arch = EinArch.detect(graph)

    assertEquals(arch.layers.size, 2)

    arch.layers(0).layerType match
      case LayerType.Dense(_, _, Some(Activation.ReLU)) => ()
      case other => fail(s"Expected Dense + ReLU for layer 0, got $other")

    arch.layers(1).layerType match
      case LayerType.Dense(_, _, None) => ()
      case other => fail(s"Expected Dense (linear) for layer 1, got $other")
  }

  // -- 8. End-to-end: block * N >> output builds and evals --

  test("end-to-end: block * N >> output builds and evals") {
    val net = (Block.dense[Double](4, Activation.Sigmoid) * 2 >> Block.linear[Double](1))
      .materialize(input2)

    // Should have 3 layers worth of params (2 dense + 1 linear)
    val params = collectParams(net)
    val paramNames = params.map(_._1).toSet
    assertEquals(paramNames, Set(
      "block0/W", "block0/b",
      "block1/W", "block1/b",
      "block2/W", "block2/b"
    ))

    // Eval should produce a scalar-like output (1 dim of size 1)
    val result = EinEval.eval(net, feedInput(1.0, 2.0))
    assertEquals(result.dims.length, 1)
    assertEquals(result.dims.head.size, 1)

    // Backward should produce gradients for all 6 params
    val grads = EinGrad.backward(net, feedInput(1.0, 2.0))
    assertEquals(grads.size, 6)
  }

  // -- 9. Pipeline syntax: inp >> block --

  test("inp >> block produces a Network that materializes correctly") {
    import Block.*
    val block = Block.dense[Double](3, Activation.ReLU)
    val net = (input2 >> block).materialize

    net match
      case Ein.Activate(Activation.ReLU, Ein.ElemAdd(Ein.Contract(w, Ein.Input("x", _)), b)) =>
        w match
          case Ein.Param("block0/W", _, _) => ()
          case other => fail(s"Expected block0/W, got $other")
      case other => fail(s"Expected Activate(ReLU, Add(Contract, bias)), got $other")
  }

  test("inp >> block1 >> block2 chains via Network") {
    import Block.*
    val b1 = Block.dense[Double](4, Activation.ReLU)
    val b2 = Block.linear[Double](1)
    val net = (input2 >> b1 >> b2).materialize

    net match
      case Ein.ElemAdd(Ein.Contract(w2, inner), _) =>
        w2 match
          case Ein.Param("block1/W", _, _) => ()
          case other => fail(s"Expected block1/W, got $other")
        inner match
          case Ein.Activate(Activation.ReLU, _) => ()
          case other => fail(s"Expected ReLU activation in inner, got $other")
      case other => fail(s"Expected linear output layer, got $other")
  }

  test("inp >> block * N >> output full pipeline syntax") {
    import Block.*
    val net = (input2 >> Block.dense[Double](4, Activation.Sigmoid) * 2 >> Block.linear[Double](1)).materialize

    val params = collectParams(net)
    val paramNames = params.map(_._1).toSet
    assertEquals(paramNames, Set(
      "block0/W", "block0/b",
      "block1/W", "block1/b",
      "block2/W", "block2/b"
    ))

    val result = EinEval.eval(net, feedInput(1.0, 2.0))
    assertEquals(result.dims.length, 1)
    assertEquals(result.dims.head.size, 1)
  }

  // -- 10. Block.embedding --

  test("embedding block materializes to Gather with correct dims") {
    val seq = Dim("seq", 3)
    val indices = TensorData.fromArray(List(seq), Array(0, 2, 1))
    val block = Block.embedding[Double](vocabSize = 5, dim = 4, indices = indices)
    val dummyInput = Ein.Input[Double]("dummy", Nil)
    val net = block.materialize(dummyInput)

    net match
      case Ein.Gather(Ein.Param(id, dims, _), idx, lookupDim) =>
        assert(id == "block0/embedding")
        assertEquals(dims, List(Dim("vocab", 5), Dim("embed", 4)))
        assertEquals(lookupDim, "vocab")
        assertEquals(idx.dims, List(seq))
      case other => fail(s"Expected Gather(Param, ...), got $other")
  }

  test("embedding block output dims are [seq, embed]") {
    val seq = Dim("seq", 3)
    val indices = TensorData.fromArray(List(seq), Array(0, 2, 1))
    val block = Block.embedding[Double](vocabSize = 5, dim = 4, indices = indices)
    val dummyInput = Ein.Input[Double]("dummy", Nil)
    val net = block.materialize(dummyInput)
    assertEquals(net.outputDims, List(seq, Dim("embed", 4)))
  }

  test("embedding block params have correct shapes") {
    val seq = Dim("seq", 3)
    val indices = TensorData.fromArray(List(seq), Array(0, 2, 1))
    val block = Block.embedding[Double](vocabSize = 5, dim = 4, indices = indices)
    val dummyInput = Ein.Input[Double]("dummy", Nil)
    val net = block.materialize(dummyInput)
    val params = collectParams(net)
    assertEquals(params.size, 1)
    assertEquals(params.head._1, "block0/embedding")
    assertEquals(params.head._2, List(Dim("vocab", 5), Dim("embed", 4)))
  }

  test("embedding block chains with downstream layers") {
    val seq = Dim("seq", 3)
    val indices = TensorData.fromArray(List(seq), Array(0, 2, 1))
    val emb = Block.embedding[Double](vocabSize = 5, dim = 4, indices = indices)
    val act = Block.activate[Double](Activation.Tanh)
    val dummyInput = Ein.Input[Double]("dummy", Nil)
    val net = (emb >> act).materialize(dummyInput)
    assertEquals(net.outputDims, List(seq, Dim("embed", 4)))
  }

  // -- helpers --

  private def collectParams(expr: Ein[Double]): List[(String, List[Dim])] =
    expr match
      case Ein.Param(id, dims, _)      => List((id, dims))
      case Ein.Input(_, _)             => Nil
      case Ein.Fill(_, _)              => Nil
      case Ein.Contract(l, r)          => collectParams(l) ++ collectParams(r)
      case Ein.ElemAdd(l, r)           => collectParams(l) ++ collectParams(r)
      case Ein.ElemSub(l, r)           => collectParams(l) ++ collectParams(r)
      case Ein.ElemMul(l, r)           => collectParams(l) ++ collectParams(r)
      case Ein.Activate(_, arg)        => collectParams(arg)
      case Ein.ActivateDeriv(_, arg)   => collectParams(arg)
      case Ein.ReduceSum(arg, _)       => collectParams(arg)
      case Ein.Broadcast(arg, _)       => collectParams(arg)
      case Ein.Transpose(arg, _)       => collectParams(arg)
      case Ein.Softmax(arg, _)         => collectParams(arg)
      case Ein.LogSoftmax(arg, _)      => collectParams(arg)
      case Ein.LayerNorm(arg, s, b, _, _) => collectParams(arg) ++ collectParams(s) ++ collectParams(b)
      case Ein.Reshape(arg, _)           => collectParams(arg)
      case Ein.Slice(arg, _, _, _)       => collectParams(arg)
      case Ein.Gather(table, _, _)       => collectParams(table)
      case Ein.Scatter(src, _, _, _)     => collectParams(src)

  private def assertArrayEquals(actual: Array[Double], expected: Array[Double]): Unit =
    assertEquals(actual.length, expected.length, s"Array length mismatch")
    actual.zip(expected).zipWithIndex.foreach { case ((a, e), i) =>
      assertEqualsDouble(a, e, 1e-10)
    }
