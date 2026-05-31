package palladium.ein

import palladium.NumberLike

class EinEvalSuite extends munit.FunSuite:

  val tolerance = 1e-9

  // Dimension helpers
  val i = Dim("i", 3)
  val j = Dim("j", 2)
  val k = Dim("k", 4)

  test("dot product: v1[i] * v2[i] = scalar") {
    val d = Dim("d", 3)
    val v1 = Ein.Param("v1", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val v2 = Ein.Param("v2", List(d), TensorData.fromArray(List(d), Array(4.0, 5.0, 6.0)))
    val dot = v1 × v2
    val result = EinEval.eval(dot)
    // 1*4 + 2*5 + 3*6 = 32
    assertEqualsDouble(result.data(0), 32.0, tolerance)
    assertEquals(result.dims, Nil)
  }

  test("matrix-vector multiply: M[i,j] * v[j] = out[i]") {
    val rows = Dim("i", 2)
    val cols = Dim("j", 3)
    // M = [[1,2,3],[4,5,6]]
    val m = Ein.Param("M", List(rows, cols),
      TensorData.fromArray(List(rows, cols), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val v = Ein.Param("v", List(cols),
      TensorData.fromArray(List(cols), Array(1.0, 0.0, 1.0)))
    val result = EinEval.eval(m × v)
    // [1+0+3, 4+0+6] = [4, 10]
    assertEquals(result.dims, List(rows))
    assertEqualsDouble(result.data(0), 4.0, tolerance)
    assertEqualsDouble(result.data(1), 10.0, tolerance)
  }

  test("matrix-matrix multiply: A[i,k] * B[k,j] = C[i,j]") {
    val mi = Dim("i", 2)
    val mk = Dim("k", 3)
    val mj = Dim("j", 2)
    // A = [[1,0,1],[0,1,0]]  (2x3)
    val a = Ein.Param("A", List(mi, mk),
      TensorData.fromArray(List(mi, mk), Array(1.0, 0.0, 1.0, 0.0, 1.0, 0.0)))
    // B = [[1,2],[3,4],[5,6]]  (3x2)
    val b = Ein.Param("B", List(mk, mj),
      TensorData.fromArray(List(mk, mj), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val result = EinEval.eval(a × b)
    assertEquals(result.dims, List(mi, mj))
    // C[0,0] = 1*1+0*3+1*5=6, C[0,1]=1*2+0*4+1*6=8
    // C[1,0] = 0*1+1*3+0*5=3, C[1,1]=0*2+1*4+0*6=4
    assertEqualsDouble(result.data(0), 6.0, tolerance)
    assertEqualsDouble(result.data(1), 8.0, tolerance)
    assertEqualsDouble(result.data(2), 3.0, tolerance)
    assertEqualsDouble(result.data(3), 4.0, tolerance)
  }

  test("outer product: v1[i] * v2[j] = M[i,j]") {
    val di = Dim("i", 2)
    val dj = Dim("j", 3)
    val v1 = Ein.Param("v1", List(di), TensorData.fromArray(List(di), Array(1.0, 2.0)))
    val v2 = Ein.Param("v2", List(dj), TensorData.fromArray(List(dj), Array(3.0, 4.0, 5.0)))
    val result = EinEval.eval(v1 × v2)
    assertEquals(result.dims, List(di, dj))
    // [[3,4,5],[6,8,10]]
    assertEqualsDouble(result.data(0), 3.0, tolerance)
    assertEqualsDouble(result.data(1), 4.0, tolerance)
    assertEqualsDouble(result.data(2), 5.0, tolerance)
    assertEqualsDouble(result.data(3), 6.0, tolerance)
    assertEqualsDouble(result.data(4), 8.0, tolerance)
    assertEqualsDouble(result.data(5), 10.0, tolerance)
  }

  test("element-wise add") {
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val b = Ein.Param("b", List(d), TensorData.fromArray(List(d), Array(10.0, 20.0, 30.0)))
    val result = EinEval.eval(a + b)
    assertEqualsDouble(result.data(0), 11.0, tolerance)
    assertEqualsDouble(result.data(1), 22.0, tolerance)
    assertEqualsDouble(result.data(2), 33.0, tolerance)
  }

  test("element-wise subtract") {
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(10.0, 20.0, 30.0)))
    val b = Ein.Param("b", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val result = EinEval.eval(a - b)
    assertEqualsDouble(result.data(0), 9.0, tolerance)
    assertEqualsDouble(result.data(1), 18.0, tolerance)
    assertEqualsDouble(result.data(2), 27.0, tolerance)
  }

  test("element-wise multiply") {
    val d = Dim("d", 3)
    val a = Ein.Param("a", List(d), TensorData.fromArray(List(d), Array(2.0, 3.0, 4.0)))
    val b = Ein.Param("b", List(d), TensorData.fromArray(List(d), Array(5.0, 6.0, 7.0)))
    val result = EinEval.eval(a.elemMul(b))
    assertEqualsDouble(result.data(0), 10.0, tolerance)
    assertEqualsDouble(result.data(1), 18.0, tolerance)
    assertEqualsDouble(result.data(2), 28.0, tolerance)
  }

  test("broadcasting: vector + scalar-like via add") {
    val d = Dim("d", 3)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(1.0, 2.0, 3.0)))
    val s = Ein.Param("s", Nil, TensorData.scalar(10.0))
    val result = EinEval.eval(v + s)
    assertEqualsDouble(result.data(0), 11.0, tolerance)
    assertEqualsDouble(result.data(1), 12.0, tolerance)
    assertEqualsDouble(result.data(2), 13.0, tolerance)
  }

  test("reduce sum over one dimension") {
    val di = Dim("i", 2)
    val dj = Dim("j", 3)
    // [[1,2,3],[4,5,6]]
    val m = Ein.Param("M", List(di, dj),
      TensorData.fromArray(List(di, dj), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val result = EinEval.eval(Ein.ReduceSum(m, List("j")))
    assertEquals(result.dims, List(di))
    // [6, 15]
    assertEqualsDouble(result.data(0), 6.0, tolerance)
    assertEqualsDouble(result.data(1), 15.0, tolerance)
  }

  test("reduce sum over all dimensions") {
    val di = Dim("i", 2)
    val dj = Dim("j", 3)
    val m = Ein.Param("M", List(di, dj),
      TensorData.fromArray(List(di, dj), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val result = EinEval.eval(Ein.ReduceSum(m, List("i", "j")))
    assertEquals(result.dims, Nil)
    assertEqualsDouble(result.data(0), 21.0, tolerance)
  }

  test("transpose") {
    val di = Dim("i", 2)
    val dj = Dim("j", 3)
    // [[1,2,3],[4,5,6]]
    val m = Ein.Param("M", List(di, dj),
      TensorData.fromArray(List(di, dj), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    val result = EinEval.eval(Ein.Transpose(m, List("j", "i")))
    assertEquals(result.dims, List(dj, di))
    // [[1,4],[2,5],[3,6]]
    assertEqualsDouble(result.data(0), 1.0, tolerance)
    assertEqualsDouble(result.data(1), 4.0, tolerance)
    assertEqualsDouble(result.data(2), 2.0, tolerance)
    assertEqualsDouble(result.data(3), 5.0, tolerance)
    assertEqualsDouble(result.data(4), 3.0, tolerance)
    assertEqualsDouble(result.data(5), 6.0, tolerance)
  }

  test("ReLU activation") {
    val d = Dim("d", 4)
    val v = Ein.Param("v", List(d),
      TensorData.fromArray(List(d), Array(-2.0, -1.0, 0.5, 3.0)))
    val result = EinEval.eval(Ein.Activate(Activation.ReLU, v))
    assertEqualsDouble(result.data(0), 0.0, tolerance)
    assertEqualsDouble(result.data(1), 0.0, tolerance)
    assertEqualsDouble(result.data(2), 0.5, tolerance)
    assertEqualsDouble(result.data(3), 3.0, tolerance)
  }

  test("sigmoid activation") {
    val d = Dim("d", 1)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(0.0)))
    val result = EinEval.eval(Ein.Activate(Activation.Sigmoid, v))
    assertEqualsDouble(result.data(0), 0.5, tolerance)
  }

  test("tanh activation") {
    val d = Dim("d", 1)
    val v = Ein.Param("v", List(d), TensorData.fromArray(List(d), Array(0.0)))
    val result = EinEval.eval(Ein.Activate(Activation.Tanh, v))
    assertEqualsDouble(result.data(0), 0.0, tolerance)
  }

  test("feed map provides data for Input") {
    val d = Dim("d", 3)
    val x = Ein.Input[Double]("x", List(d))
    val w = Ein.Param("w", List(d), TensorData.fromArray(List(d), Array(2.0, 3.0, 4.0)))
    val expr = x × w // dot product
    val feed = Map("x" -> TensorData.fromArray(List(d), Array(1.0, 1.0, 1.0)))
    val result = EinEval.eval(expr, feed)
    // 2+3+4 = 9
    assertEqualsDouble(result.data(0), 9.0, tolerance)
  }

  test("dense layer forward: relu(W[i,j] * x[j] + b[i])") {
    val features = Dim("j", 3)
    val hidden = Dim("i", 2)
    // W = [[1,0,0],[0,1,0]] (identity-like, 2x3)
    val w = Ein.Param("W", List(hidden, features),
      TensorData.fromArray(List(hidden, features), Array(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)))
    // b = [-0.5, 0.5]
    val b = Ein.Param("b", List(hidden),
      TensorData.fromArray(List(hidden), Array(-0.5, 0.5)))
    // x = [1, 2, 3]
    val x = Ein.Param("x", List(features),
      TensorData.fromArray(List(features), Array(1.0, 2.0, 3.0)))
    val out = Ein.Activate(Activation.ReLU, (w × x) + b)
    val result = EinEval.eval(out)
    // W*x = [1, 2], +b = [0.5, 2.5], relu = [0.5, 2.5]
    assertEqualsDouble(result.data(0), 0.5, tolerance)
    assertEqualsDouble(result.data(1), 2.5, tolerance)
  }

  test("Fill creates constant tensor") {
    val d = Dim("d", 3)
    val fill = Ein.Fill(5.0, List(d))
    val result = EinEval.eval(fill)
    assertEqualsDouble(result.data(0), 5.0, tolerance)
    assertEqualsDouble(result.data(1), 5.0, tolerance)
    assertEqualsDouble(result.data(2), 5.0, tolerance)
  }

  test("broadcast scalar to vector") {
    val d = Dim("d", 3)
    val s = Ein.Param("s", Nil, TensorData.scalar(7.0))
    val result = EinEval.eval(Ein.Broadcast(s, List(d)))
    assertEquals(result.dims, List(d))
    assertEqualsDouble(result.data(0), 7.0, tolerance)
    assertEqualsDouble(result.data(1), 7.0, tolerance)
    assertEqualsDouble(result.data(2), 7.0, tolerance)
  }
