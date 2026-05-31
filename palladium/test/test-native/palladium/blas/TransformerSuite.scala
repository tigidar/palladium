package palladium.blas

import palladium.codegen.*
import palladium.ein.*

/** Tests that transformer models run correctly through the BLAS backend,
  * comparing results against EinEval (the reference interpreter).
  */
class TransformerSuite extends munit.FunSuite:

  val eps = 1e-6

  private def assertBlasMatchesEval(
      expr: Ein[Double],
      feed: Map[String, TensorData[Double]],
      tolerance: Double = eps
  ): Unit =
    val expected = EinEval.eval(expr, feed)
    val actual = BlasGen.eval(expr, feed)

    assertEquals(actual.data.length, expected.data.length,
      s"Output size mismatch: got ${actual.data.length}, expected ${expected.data.length}")

    var i = 0
    while i < actual.data.length do
      assertEqualsDouble(actual.data(i), expected.data(i), tolerance,
        s"Mismatch at index $i: got ${actual.data(i)}, expected ${expected.data(i)}")
      i += 1

  test("single-head attention via BLAS matches EinEval") {
    val seq = Dim("seq", 3)
    val model = Dim("model", 4)
    val input = Ein.Input[Double]("x", List(seq, model))

    val attnBlock = Block.attention[Double](4)
    val output = attnBlock.materialize(input)

    val xData = TensorData.fromArray(List(seq, model),
      Array.tabulate(12)(i => (i + 1).toDouble / 12.0))
    assertBlasMatchesEval(output, Map("x" -> xData))
  }

  test("multi-head attention via BLAS matches EinEval") {
    val seq = Dim("seq", 3)
    val model = Dim("model", 4)
    val input = Ein.Input[Double]("x", List(seq, model))

    val mhaBlock = Block.multiHeadAttention[Double](nHeads = 2, headDim = 2)
    val output = mhaBlock.materialize(input)

    val xData = TensorData.fromArray(List(seq, model),
      Array.tabulate(12)(i => (i + 1).toDouble / 12.0))
    assertBlasMatchesEval(output, Map("x" -> xData))
  }

  test("feed-forward block via BLAS matches EinEval") {
    val seq = Dim("seq", 3)
    val model = Dim("model", 4)
    val input = Ein.Input[Double]("x", List(seq, model))

    val ffBlock = Block.feedForward[Double](ffDim = 8)
    val output = ffBlock.materialize(input)

    val xData = TensorData.fromArray(List(seq, model),
      Array.tabulate(12)(i => (i + 1).toDouble / 12.0))
    assertBlasMatchesEval(output, Map("x" -> xData))
  }

  test("transformer block via BLAS matches EinEval") {
    val seq = Dim("seq", 3)
    val model = Dim("model", 4)
    val input = Ein.Input[Double]("x", List(seq, model))

    val block = Block.transformerBlock[Double](
      nHeads = 2, headDim = 2, ffDim = 8, seqLen = 3
    )
    val output = block.materialize(input)

    val xData = TensorData.fromArray(List(seq, model),
      Array.tabulate(12)(i => (i + 1).toDouble / 12.0))
    assertBlasMatchesEval(output, Map("x" -> xData), tolerance = 1e-5)
  }

  test("2-layer transformer via BLAS matches EinEval") {
    val seq = Dim("seq", 3)
    val model = Dim("model", 4)
    val input = Ein.Input[Double]("x", List(seq, model))

    val transformer = Block.transformer[Double](
      nLayers = 2, nHeads = 2, headDim = 2, ffDim = 8, seqLen = 3
    )
    val output = transformer.materialize(input)

    val xData = TensorData.fromArray(List(seq, model),
      Array.tabulate(12)(i => (i + 1).toDouble / 12.0))
    assertBlasMatchesEval(output, Map("x" -> xData), tolerance = 1e-4)
  }
