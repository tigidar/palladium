package palladium.blas

import scala.scalanative.unsafe.*

/** Tests for NativeTensor arena-based tensor allocation. */
class NativeTensorSuite extends munit.FunSuite:

  val eps = 1e-10

  test("zeros: allocates zero-filled tensor") {
    Zone.acquire { implicit z =>
      val t = NativeTensor.zeros(List(3, 4))
      assertEquals(t.totalSize, 12)
      assertEquals(t.shape, List(3, 4))
      var i = 0
      while i < 12 do
        assertEqualsDouble(t(i), 0.0, eps)
        i += 1
    }
  }

  test("ones: allocates ones-filled tensor") {
    Zone.acquire { implicit z =>
      val t = NativeTensor.ones(List(2, 3))
      assertEquals(t.totalSize, 6)
      var i = 0
      while i < 6 do
        assertEqualsDouble(t(i), 1.0, eps)
        i += 1
    }
  }

  test("fromArray: wraps existing data") {
    Zone.acquire { implicit z =>
      val data = Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
      val t = NativeTensor.fromArray(List(2, 3), data)
      assertEqualsDouble(t(0), 1.0, eps)
      assertEqualsDouble(t(3), 4.0, eps)
      assertEqualsDouble(t(5), 6.0, eps)
    }
  }

  test("toArray: extracts data back to Scala Array") {
    Zone.acquire { implicit z =>
      val data = Array(1.0, 2.0, 3.0)
      val t = NativeTensor.fromArray(List(3), data)
      val out = t.toArray
      assertEquals(out.toList, List(1.0, 2.0, 3.0))
    }
  }

  test("update and read back") {
    Zone.acquire { implicit z =>
      val t = NativeTensor.zeros(List(5))
      t(2) = 42.0
      assertEqualsDouble(t(2), 42.0, eps)
      assertEqualsDouble(t(0), 0.0, eps)
    }
  }

  test("zero: clears existing data") {
    Zone.acquire { implicit z =>
      val t = NativeTensor.ones(List(4))
      t.zero()
      var i = 0
      while i < 4 do
        assertEqualsDouble(t(i), 0.0, eps)
        i += 1
    }
  }

  test("copyFrom: copies data from another tensor") {
    Zone.acquire { implicit z =>
      val src = NativeTensor.fromArray(List(3), Array(1.0, 2.0, 3.0))
      val dst = NativeTensor.zeros(List(3))
      dst.copyFrom(src)
      assertEqualsDouble(dst(0), 1.0, eps)
      assertEqualsDouble(dst(1), 2.0, eps)
      assertEqualsDouble(dst(2), 3.0, eps)
    }
  }

  test("filled: allocates with constant value") {
    Zone.acquire { implicit z =>
      val t = NativeTensor.filled(List(3), 7.5)
      assertEqualsDouble(t(0), 7.5, eps)
      assertEqualsDouble(t(1), 7.5, eps)
      assertEqualsDouble(t(2), 7.5, eps)
    }
  }

  test("scalar tensor (empty shape)") {
    Zone.acquire { implicit z =>
      val t = NativeTensor.zeros(List())
      assertEquals(t.totalSize, 1)
      t(0) = 3.14
      assertEqualsDouble(t(0), 3.14, eps)
    }
  }

  test("arena cleanup: tensors allocated in Zone are scoped") {
    // This test verifies the arena pattern works — tensors are
    // allocated inside Zone.acquire and cannot escape safely.
    // We just verify the pattern compiles and runs without error.
    var size = 0
    Zone.acquire { implicit z =>
      val t = NativeTensor.zeros(List(1000))
      size = t.totalSize
    }
    assertEquals(size, 1000)
  }
