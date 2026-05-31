package palladium.ein

class PositionalEncodingSuite extends munit.FunSuite:

  val tolerance = 1e-4 // approxFromDouble introduces small rounding

  test("output shape is [maxLen, dim]") {
    val pe = PositionalEncoding.sinusoidal[Double](maxLen = 10, dim = 8)
    assertEquals(pe.dims, List(Dim("seq", 10), Dim("embed", 8)))
    assertEquals(pe.data.length, 80)
  }

  test("PE(0, 0) = sin(0) = 0") {
    val pe = PositionalEncoding.sinusoidal[Double](maxLen = 5, dim = 4)
    // pos=0, i=0 (even): sin(0 / 10000^0) = sin(0) = 0
    assertEqualsDouble(pe.data(0), 0.0, tolerance)
  }

  test("PE(0, 1) = cos(0) = 1") {
    val pe = PositionalEncoding.sinusoidal[Double](maxLen = 5, dim = 4)
    // pos=0, i=1 (odd): cos(0 / 10000^0) = cos(0) = 1
    assertEqualsDouble(pe.data(1), 1.0, tolerance)
  }

  test("even indices use sin, odd indices use cos") {
    val pe = PositionalEncoding.sinusoidal[Double](maxLen = 3, dim = 6)
    val dim = 6
    // For pos=1, verify sin/cos pattern
    for i <- 0 until dim do
      val divTerm = math.pow(10000.0, (i / 2) * 2.0 / dim)
      val angle = 1.0 / divTerm
      val expected = if i % 2 == 0 then math.sin(angle) else math.cos(angle)
      assertEqualsDouble(pe.data(1 * dim + i), expected, tolerance,
        s"Mismatch at pos=1, i=$i")
  }

  test("values match formula for specific positions") {
    val pe = PositionalEncoding.sinusoidal[Double](maxLen = 10, dim = 8)
    val dim = 8
    for pos <- List(0, 3, 7) do
      for i <- List(0, 1, 4, 5) do
        val divTerm = math.pow(10000.0, (i / 2) * 2.0 / dim)
        val angle = pos.toDouble / divTerm
        val expected = if i % 2 == 0 then math.sin(angle) else math.cos(angle)
        assertEqualsDouble(pe.data(pos * dim + i), expected, tolerance,
          s"Mismatch at pos=$pos, i=$i")
  }

  test("different positions produce different encodings") {
    val pe = PositionalEncoding.sinusoidal[Double](maxLen = 5, dim = 4)
    val dim = 4
    val row0 = pe.data.slice(0, dim)
    val row1 = pe.data.slice(dim, 2 * dim)
    assert(!row0.zip(row1).forall((a, b) => math.abs(a - b) < 1e-10),
      "Different positions should have different encodings")
  }

  test("custom dim names") {
    val pe = PositionalEncoding.sinusoidal[Double](
      maxLen = 5, dim = 4,
      posDimName = "time", embedDimName = "hidden"
    )
    assertEquals(pe.dims, List(Dim("time", 5), Dim("hidden", 4)))
  }

  test("works with Float type") {
    val pe = PositionalEncoding.sinusoidal[Float](maxLen = 3, dim = 4)
    assertEquals(pe.dims, List(Dim("seq", 3), Dim("embed", 4)))
    assertEquals(pe.data.length, 12)
    // PE(0,0) = sin(0) = 0
    assert(math.abs(pe.data(0)) < 0.001f)
    // PE(0,1) = cos(0) = 1
    assert(math.abs(pe.data(1) - 1.0f) < 0.001f)
  }

  test("can be used as Ein.Param for tensor operations") {
    val pe = PositionalEncoding.sinusoidal[Double](maxLen = 4, dim = 6)
    val peParam = Ein.Param("pos_enc", pe.dims, pe)
    assertEquals(peParam.outputDims, List(Dim("seq", 4), Dim("embed", 6)))
    // Should be evaluable
    val result = EinEval.eval(peParam)
    assertEquals(result.data.length, 24)
  }
