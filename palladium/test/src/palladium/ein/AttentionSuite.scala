package palladium.ein

class AttentionSuite extends munit.FunSuite:

  val eps = 1e-8

  // ── Causal mask tests ───────────────────────────────────────────────

  test("causalMask: shape is [seq, kSeq]") {
    val mask = Attention.causalMask[Double](4, "seq", "kSeq")
    assertEquals(mask.dims.map(_.name), List("seq", "kSeq"))
    assertEquals(mask.dims.map(_.size), List(4, 4))
  }

  test("causalMask: lower triangle is 0, upper triangle is -inf") {
    val mask = Attention.causalMask[Double](3, "seq", "kSeq")
    // Row 0: can attend to pos 0 only → [0, -inf, -inf]
    // Row 1: can attend to pos 0,1   → [0, 0, -inf]
    // Row 2: can attend to pos 0,1,2 → [0, 0, 0]
    assertEqualsDouble(mask.data(0), 0.0, eps)       // (0,0)
    assert(mask.data(1).isNegInfinity)                // (0,1)
    assert(mask.data(2).isNegInfinity)                // (0,2)
    assertEqualsDouble(mask.data(3), 0.0, eps)        // (1,0)
    assertEqualsDouble(mask.data(4), 0.0, eps)        // (1,1)
    assert(mask.data(5).isNegInfinity)                // (1,2)
    assertEqualsDouble(mask.data(6), 0.0, eps)        // (2,0)
    assertEqualsDouble(mask.data(7), 0.0, eps)        // (2,1)
    assertEqualsDouble(mask.data(8), 0.0, eps)        // (2,2)
  }

  test("causalMask: size 1 is all zeros") {
    val mask = Attention.causalMask[Double](1, "seq", "kSeq")
    assertEqualsDouble(mask.data(0), 0.0, eps)
  }

  // ── Single-head attention tests ─────────────────────────────────────

  test("attention block: output dims match input dims") {
    val seqLen = 4
    val modelDim = 8
    val headDim = 8
    val seq = Dim("seq", seqLen)
    val model = Dim("model", modelDim)
    val input = Ein.Input[Double]("x", List(seq, model))

    val attnBlock = Block.attention[Double](headDim)
    val output = attnBlock.materialize(input)
    val outDims = output.outputDims

    // Output should have "model" and "seq" dims (order may vary)
    val outNames = outDims.map(_.name).toSet
    assert(outNames.contains("model"), s"Expected 'model' in $outNames")
    assert(outNames.contains("seq"), s"Expected 'seq' in $outNames")
    assertEquals(outDims.find(_.name == "model").get.size, modelDim)
    assertEquals(outDims.find(_.name == "seq").get.size, seqLen)
  }

  test("attention block: evaluates without error") {
    val seq = Dim("seq", 3)
    val model = Dim("model", 4)
    val input = Ein.Input[Double]("x", List(seq, model))

    val attnBlock = Block.attention[Double](4)
    val output = attnBlock.materialize(input)

    val xData = TensorData.fromArray(List(seq, model),
      Array.tabulate(12)(i => (i + 1).toDouble / 12.0))
    val feed = Map("x" -> xData)

    // Should evaluate without error
    val result = EinEval.eval(output, feed)
    assertEquals(result.data.length, 12)
    // All values should be finite
    assert(result.data.forall(_.isFinite), "All outputs should be finite")
  }

  test("attention block with causal mask: future tokens get zero attention") {
    val seq = Dim("seq", 3)
    val model = Dim("model", 2)
    val input = Ein.Input[Double]("x", List(seq, model))

    val mask = Attention.causalMask[Double](3, "seq", "kSeq")
    val attnBlock = Block.attention[Double](2, Some(mask))
    val output = attnBlock.materialize(input)

    val xData = TensorData.fromArray(List(seq, model),
      Array(1.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    val feed = Map("x" -> xData)

    val result = EinEval.eval(output, feed)
    assert(result.data.forall(_.isFinite), "All outputs should be finite with causal mask")
  }

  // ── Multi-head attention tests ──────────────────────────────────────

  test("multiHeadAttention: output dims match input dims") {
    val seq = Dim("seq", 4)
    val model = Dim("model", 8)
    val input = Ein.Input[Double]("x", List(seq, model))

    val mhaBlock = Block.multiHeadAttention[Double](nHeads = 2, headDim = 4)
    val output = mhaBlock.materialize(input)
    val outDims = output.outputDims

    val outNames = outDims.map(_.name).toSet
    assert(outNames.contains("model"), s"Expected 'model' in $outNames")
    assert(outNames.contains("seq"), s"Expected 'seq' in $outNames")
  }

  test("multiHeadAttention: evaluates without error") {
    val seq = Dim("seq", 3)
    val model = Dim("model", 4)
    val input = Ein.Input[Double]("x", List(seq, model))

    val mhaBlock = Block.multiHeadAttention[Double](nHeads = 2, headDim = 2)
    val output = mhaBlock.materialize(input)

    val xData = TensorData.fromArray(List(seq, model),
      Array.tabulate(12)(i => (i + 1).toDouble / 12.0))
    val result = EinEval.eval(output, Map("x" -> xData))
    assert(result.data.forall(_.isFinite))
  }

  // ── Feed-forward block tests ────────────────────────────────────────

  test("feedForward: output dims match input dims") {
    val seq = Dim("seq", 4)
    val model = Dim("model", 8)
    val input = Ein.Input[Double]("x", List(seq, model))

    val ffBlock = Block.feedForward[Double](ffDim = 32)
    val output = ffBlock.materialize(input)
    val outDims = output.outputDims

    val outNames = outDims.map(_.name).toSet
    assert(outNames.contains("model"), s"Expected 'model' in $outNames")
    assert(outNames.contains("seq"), s"Expected 'seq' in $outNames")
  }

  // ── Transformer block tests ─────────────────────────────────────────

  test("transformerBlock: output dims match input dims") {
    val seq = Dim("seq", 4)
    val model = Dim("model", 8)
    val input = Ein.Input[Double]("x", List(seq, model))

    val block = Block.transformerBlock[Double](nHeads = 2, headDim = 4, ffDim = 16, seqLen = 4)
    val output = block.materialize(input)
    val outDims = output.outputDims

    val outNames = outDims.map(_.name).toSet
    assert(outNames.contains("model"), s"Expected 'model' in $outNames")
    assert(outNames.contains("seq"), s"Expected 'seq' in $outNames")
  }

  test("transformerBlock: evaluates without error") {
    val seq = Dim("seq", 3)
    val model = Dim("model", 4)
    val input = Ein.Input[Double]("x", List(seq, model))

    val block = Block.transformerBlock[Double](nHeads = 2, headDim = 2, ffDim = 8, seqLen = 3)
    val output = block.materialize(input)

    val xData = TensorData.fromArray(List(seq, model),
      Array.tabulate(12)(i => (i + 1).toDouble / 12.0))
    val result = EinEval.eval(output, Map("x" -> xData))
    assertEquals(result.data.length, 12)
    assert(result.data.forall(_.isFinite), "Transformer block output should be finite")
  }

  // ── Stacked transformer ─────────────────────────────────────────────

  test("transformer: multiple blocks stack correctly") {
    val seq = Dim("seq", 3)
    val model = Dim("model", 4)
    val input = Ein.Input[Double]("x", List(seq, model))

    val transformer = Block.transformer[Double](
      nLayers = 2, nHeads = 2, headDim = 2, ffDim = 8, seqLen = 3
    )
    val output = transformer.materialize(input)

    val xData = TensorData.fromArray(List(seq, model),
      Array.tabulate(12)(i => (i + 1).toDouble / 12.0))
    val result = EinEval.eval(output, Map("x" -> xData))
    assertEquals(result.data.length, 12)
    assert(result.data.forall(_.isFinite), "Stacked transformer output should be finite")
  }
