package palladium.ein

class TensorDataSuite extends munit.FunSuite:

  val tolerance = 1e-10

  // ============================================
  // Factory methods
  // ============================================

  test("zeros creates all-zero tensor with correct dims") {
    val dims = List(Dim("i", 2), Dim("j", 3))
    val t = TensorData.zeros[Double](dims)
    assertEquals(t.dims, dims)
    assertEquals(t.size, 6)
    for i <- 0 until 6 do
      assertEqualsDouble(t.data(i), 0.0, tolerance)
  }

  test("fill creates tensor with uniform value") {
    val dims = List(Dim("x", 3))
    val t = TensorData.fill[Double](dims, 5.0)
    assertEquals(t.size, 3)
    for i <- 0 until 3 do
      assertEqualsDouble(t.data(i), 5.0, tolerance)
  }

  test("fromArray creates tensor from provided data") {
    val dims = List(Dim("r", 2), Dim("c", 2))
    val t = TensorData.fromArray[Double](dims, Array(1.0, 2.0, 3.0, 4.0))
    assertEqualsDouble(t.data(0), 1.0, tolerance)
    assertEqualsDouble(t.data(3), 4.0, tolerance)
  }

  test("fromArray clones the input array") {
    val dims = List(Dim("d", 3))
    val original = Array(1.0, 2.0, 3.0)
    val t = TensorData.fromArray[Double](dims, original)
    original(0) = 999.0
    assertEqualsDouble(t.data(0), 1.0, tolerance, "TensorData should not alias input array")
  }

  test("scalar creates rank-0 tensor") {
    val t = TensorData.scalar[Double](42.0)
    assertEquals(t.dims, Nil)
    assertEquals(t.size, 1)
    assertEqualsDouble(t.data(0), 42.0, tolerance)
  }

  // ============================================
  // Strides and indexing
  // ============================================

  test("strides are row-major") {
    val dims = List(Dim("i", 2), Dim("j", 3))
    val strides = TensorData.computeStrides(dims)
    assertEquals(strides, List(3, 1))
  }

  test("strides for 3D tensor") {
    val dims = List(Dim("i", 2), Dim("j", 3), Dim("k", 4))
    val strides = TensorData.computeStrides(dims)
    assertEquals(strides, List(12, 4, 1))
  }

  test("strides for scalar are empty") {
    assertEquals(TensorData.computeStrides(Nil), Nil)
  }

  test("offset computes correct flat index") {
    val dims = List(Dim("i", 2), Dim("j", 3))
    val t = TensorData.fromArray[Double](dims, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    // Row-major: t(0,0)=1, t(0,1)=2, t(0,2)=3, t(1,0)=4, t(1,1)=5, t(1,2)=6
    assertEqualsDouble(t(Array(0, 0)), 1.0, tolerance)
    assertEqualsDouble(t(Array(0, 2)), 3.0, tolerance)
    assertEqualsDouble(t(Array(1, 0)), 4.0, tolerance)
    assertEqualsDouble(t(Array(1, 2)), 6.0, tolerance)
  }

  test("update modifies tensor in place") {
    val dims = List(Dim("d", 3))
    val t = TensorData.zeros[Double](dims)
    t.update(Array(1), 7.0)
    assertEqualsDouble(t(Array(1)), 7.0, tolerance)
  }

  // ============================================
  // Equality and hashing
  // ============================================

  test("equals compares dims and data") {
    val dims = List(Dim("d", 2))
    val a = TensorData.fromArray[Double](dims, Array(1.0, 2.0))
    val b = TensorData.fromArray[Double](dims, Array(1.0, 2.0))
    val c = TensorData.fromArray[Double](dims, Array(1.0, 3.0))
    assertEquals(a, b)
    assertNotEquals(a, c)
  }

  test("equals returns false for different dims") {
    val a = TensorData.fromArray[Double](List(Dim("x", 2)), Array(1.0, 2.0))
    val b = TensorData.fromArray[Double](List(Dim("y", 2)), Array(1.0, 2.0))
    assertNotEquals(a, b)
  }

  test("equals returns false for non-TensorData") {
    val t = TensorData.scalar[Double](1.0)
    assert(!t.equals("not a tensor"))
  }

  test("hashCode is consistent with equals") {
    val dims = List(Dim("d", 3))
    val a = TensorData.fromArray[Double](dims, Array(1.0, 2.0, 3.0))
    val b = TensorData.fromArray[Double](dims, Array(1.0, 2.0, 3.0))
    assertEquals(a.hashCode, b.hashCode)
  }

  // ============================================
  // toString
  // ============================================

  test("toString includes dim info") {
    val dims = List(Dim("i", 2), Dim("j", 3))
    val t = TensorData.zeros[Double](dims)
    val s = t.toString
    assert(s.contains("i=2"), s"Expected 'i=2' in $s")
    assert(s.contains("j=3"), s"Expected 'j=3' in $s")
  }

  test("toString truncates large tensors") {
    val dims = List(Dim("d", 20))
    val t = TensorData.fill[Double](dims, 1.0)
    val s = t.toString
    assert(s.contains("..."), s"Expected '...' in $s")
  }

  // ============================================
  // Edge cases
  // ============================================

  test("zeros with empty dims creates scalar") {
    val t = TensorData.zeros[Double](Nil)
    assertEquals(t.size, 1)
    assertEqualsDouble(t.data(0), 0.0, tolerance)
  }

  test("zeros with single dim of size 1") {
    val t = TensorData.zeros[Double](List(Dim("x", 1)))
    assertEquals(t.size, 1)
  }

  test("Float tensor operations") {
    val dims = List(Dim("d", 2))
    val t = TensorData.fill[Float](dims, 3.14f)
    assertEquals(t.size, 2)
    assertEquals(t.data(0), 3.14f)
  }
