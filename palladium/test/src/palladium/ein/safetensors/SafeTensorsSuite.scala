package palladium.ein.safetensors

import munit.FunSuite
import palladium.ein.{Dim, Ein, TensorData}
import palladium.NumberLike.given
import scodec.bits.*

/** Integration tests for the palladium adapter over `no.virtual_architect.safetensors`.
  *
  * Header parsing, dtype decoding, and raw read/loadAllFloat/loadAllDouble
  * are covered by the upstream `safetensors-scala` suite. This suite covers
  * only the palladium-specific surface: named-dim wrapping (`TensorData[A]`)
  * and `loadWeights` substitution into Ein expression trees.
  */
class SafeTensorsSuite extends FunSuite:

  def floatsToBytes(values: Float*): Array[Byte] =
    values.foldLeft(ByteVector.empty) { (acc, v) =>
      acc ++ ByteVector.fromInt(java.lang.Float.floatToIntBits(v), ordering = ByteOrdering.LittleEndian)
    }.toArray

  def doublesToBytes(values: Double*): Array[Byte] =
    values.foldLeft(ByteVector.empty) { (acc, v) =>
      acc ++ ByteVector.fromLong(java.lang.Double.doubleToLongBits(v), ordering = ByteOrdering.LittleEndian)
    }.toArray

  def buildSafeTensors(tensors: (String, DType, List[Int], Array[Byte])*): Array[Byte] =
    var offset = 0L
    val withOffsets = tensors.map { (name, dtype, shape, data) =>
      val start = offset
      offset += data.length
      (name, dtype, shape, data, start, offset)
    }
    val entries = withOffsets.map { (name, dtype, shape, _, start, end) =>
      val shapeStr = shape.mkString("[", ",", "]")
      s""""$name":{"dtype":"${dtype.toString}","shape":$shapeStr,"data_offsets":[$start,$end]}"""
    }
    val headerJson = "{" + entries.mkString(",") + "}"
    val headerBv = ByteVector.encodeAscii(headerJson).fold(e => throw RuntimeException(e.toString), identity)
    val headerSizeBv = ByteVector.fromLong(headerBv.size, ordering = ByteOrdering.LittleEndian)
    val dataBv = tensors.foldLeft(ByteVector.empty) { (acc, t) => acc ++ ByteVector(t._4) }
    (headerSizeBv ++ headerBv ++ dataBv).toArray

  // ── Adapter: named-dim wrapping ──────────────────────────────────────

  test("adapter readFloat wraps shape with caller-supplied dim names") {
    val bytes = buildSafeTensors(
      ("W", DType.F32, List(3, 2), floatsToBytes(1f, 2f, 3f, 4f, 5f, 6f))
    )
    val header = SafeTensors.parseHeader(bytes)
    val td = SafeTensors.readFloat(bytes, header, "W", List("out", "inp"))

    assertEquals(td.dims, List(Dim("out", 3), Dim("inp", 2)))
    assertEquals(td.data.length, 6)
    assertEqualsFloat(td.data(0), 1f, 1e-6f)
  }

  test("adapter readFloat fails for mismatched dim count") {
    val bytes = buildSafeTensors(
      ("x", DType.F32, List(3, 2), floatsToBytes(1f, 2f, 3f, 4f, 5f, 6f))
    )
    val header = SafeTensors.parseHeader(bytes)
    val ex = intercept[IllegalArgumentException] {
      SafeTensors.readFloat(bytes, header, "x", List("d"))
    }
    assert(ex.getMessage.contains("dimNames length"), s"Expected 'dimNames length' in: ${ex.getMessage}")
  }

  test("adapter loadAll loads all tensors as Double with caller-supplied dim names") {
    val bytes = buildSafeTensors(
      ("W", DType.F32, List(3, 2), floatsToBytes(1f, 2f, 3f, 4f, 5f, 6f)),
      ("b", DType.F32, List(3), floatsToBytes(0.1f, 0.2f, 0.3f))
    )
    val dimMapping: String => List[String] = {
      case "W" => List("hid", "inp")
      case "b" => List("hid")
    }
    val loaded = SafeTensors.loadAll[Double](bytes, dimMapping)
    assertEquals(loaded.size, 2)
    assertEquals(loaded("W").dims, List(Dim("hid", 3), Dim("inp", 2)))
    assertEquals(loaded("b").dims, List(Dim("hid", 3)))
    assertEqualsDouble(loaded("W").data(0), 1.0, 1e-6)
  }

  // ── loadWeights integration ──────────────────────────────────────────

  test("loadWeights replaces Ein.Param data in expression tree") {
    val hid = Dim("hid", 3)
    val inp = Dim("inp", 2)

    val w1 = Ein.Param("W1", List(hid, inp), TensorData.zeros[Double](List(hid, inp)))
    val b1 = Ein.Param("b1", List(hid), TensorData.zeros[Double](List(hid)))
    val x = Ein.Input[Double]("x", List(inp))
    val expr = (w1 * x) + b1

    val bytes = buildSafeTensors(
      ("W1", DType.F64, List(3, 2), doublesToBytes(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
      ("b1", DType.F64, List(3), doublesToBytes(0.1, 0.2, 0.3))
    )
    val header = SafeTensors.parseHeader(bytes)

    val w1Loaded = SafeTensors.readDouble(bytes, header, "W1", List("hid", "inp"))
    val b1Loaded = SafeTensors.readDouble(bytes, header, "b1", List("hid"))
    val weights = Map("W1" -> w1Loaded, "b1" -> b1Loaded)
    val loaded = SafeTensors.loadWeights(expr, weights)

    def findParam(e: Ein[Double], name: String): Option[TensorData[Double]] =
      e match
        case Ein.Param(id, _, data) if id == name => Some(data)
        case Ein.Contract(l, r) => findParam(l, name).orElse(findParam(r, name))
        case Ein.ElemAdd(l, r) => findParam(l, name).orElse(findParam(r, name))
        case _ => None

    val w1Data = findParam(loaded, "W1").get
    assertEqualsDouble(w1Data.data(0), 1.0, 1e-12)
    assertEqualsDouble(w1Data.data(5), 6.0, 1e-12)

    val b1Data = findParam(loaded, "b1").get
    assertEqualsDouble(b1Data.data(0), 0.1, 1e-12)
    assertEqualsDouble(b1Data.data(2), 0.3, 1e-12)
  }

  test("loadWeights leaves unmatched params unchanged") {
    val d = Dim("d", 2)
    val w = Ein.Param("W", List(d), TensorData.zeros[Double](List(d)))
    val loaded = SafeTensors.loadWeights(w, Map.empty)

    loaded match
      case Ein.Param(_, _, data) =>
        assertEqualsDouble(data.data(0), 0.0, 1e-12)
      case _ => fail("Expected Param")
  }
