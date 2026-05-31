package palladium.codegen

import palladium.ein.*

class LowerSuite extends munit.FunSuite:

  test("lower single param produces one param, no ops") {
    val x = Ein.Param[Double]("x", List(Dim("d", 3)), TensorData.zeros(List(Dim("d", 3))))
    val prog = Lower.lower(x)

    assertEquals(prog.params.size, 1)
    assertEquals(prog.params.head._2, "x")
    assertEquals(prog.params.head._1.shape, List(3))
    assertEquals(prog.ops.size, 0)
  }

  test("lower input produces one input") {
    val x = Ein.Input[Double]("x", List(Dim("d", 3)))
    val prog = Lower.lower(x)

    assertEquals(prog.inputs.size, 1)
    assertEquals(prog.inputs.head._2, "x")
  }

  test("lower contract (matrix-vector multiply)") {
    val inp = Dim("inp", 2)
    val hid = Dim("hid", 3)
    val w = Ein.Param[Double]("W", List(hid, inp), TensorData.zeros(List(hid, inp)))
    val x = Ein.Input[Double]("x", List(inp))
    val expr = w * x

    val prog = Lower.lower(expr)

    assertEquals(prog.params.size, 1)
    assertEquals(prog.inputs.size, 1)
    assert(prog.ops.exists {
      case LowOp.MatMul(_, _, _, _, _, _, _) => true
      case _ => false
    })
    assertEquals(prog.outputs.head.shape, List(3))
  }

  test("lower 2-layer MLP") {
    val inp = Dim("inp", 2)
    val hid = Dim("hid", 3)
    val out = Dim("out", 1)

    val w1 = Ein.Param[Double]("W1", List(hid, inp), TensorData.zeros(List(hid, inp)))
    val b1 = Ein.Param[Double]("b1", List(hid), TensorData.zeros(List(hid)))
    val w2 = Ein.Param[Double]("W2", List(out, hid), TensorData.zeros(List(out, hid)))
    val b2 = Ein.Param[Double]("b2", List(out), TensorData.zeros(List(out)))
    val x = Ein.Input[Double]("x", List(inp))

    val h = Ein.Activate(Activation.ReLU, (w1 * x) + b1)
    val output = (w2 * h) + b2

    val prog = Lower.lower(output)

    assertEquals(prog.params.size, 4)
    assertEquals(prog.inputs.size, 1)

    val matMuls = prog.ops.collect { case m: LowOp.MatMul => m }
    val adds = prog.ops.collect { case a: LowOp.ElemBinary => a }
    val activations = prog.ops.collect { case a: LowOp.Activate => a }

    assertEquals(matMuls.size, 2)
    assertEquals(adds.size, 2)
    assertEquals(activations.size, 1)
    assertEquals(prog.outputs.head.shape, List(1))
  }

  test("lower block-generated network") {
    val inp2 = Dim("inp", 2)
    val input = Ein.Input[Double]("x", List(inp2))

    val net = (Block.dense[Double](4, Activation.ReLU) >> Block.linear[Double](1))
      .materialize(input)

    val prog = Lower.lower(net)

    assertEquals(prog.params.size, 4)
    assertEquals(prog.inputs.size, 1)
    assertEquals(prog.outputs.head.shape, List(1))
  }

  test("lower Ones/Zeros nodes") {
    val dims = List(Dim("d", 3))
    val ones = Ein.Ones[Double](dims)
    val prog = Lower.lower(ones)

    assert(prog.ops.exists {
      case LowOp.FillOnes(_) => true
      case _ => false
    })
  }

  test("lower elem-wise add") {
    val d = Dim("d", 3)
    val a = Ein.Input[Double]("a", List(d))
    val b = Ein.Input[Double]("b", List(d))
    val expr = a + b

    val prog = Lower.lower(expr)

    assertEquals(prog.inputs.size, 2)
    assert(prog.ops.exists {
      case LowOp.ElemBinary(_, _, _, BinaryOp.Add) => true
      case _ => false
    })
    assertEquals(prog.outputs.head.shape, List(3))
  }

  test("param deduplication: same Param object produces same TensorRef") {
    val d = Dim("d", 3)
    val w = Ein.Param[Double]("W", List(d), TensorData.zeros(List(d)))
    val expr = w + w

    val prog = Lower.lower(expr)

    assertEquals(prog.params.size, 1)
  }

  test("lower softmax") {
    val d = Dim("d", 4)
    val x = Ein.Input[Double]("x", List(d))
    val expr = Ein.Softmax(x, "d")

    val prog = Lower.lower(expr)

    assert(prog.ops.exists {
      case LowOp.Softmax(_, _, _) => true
      case _ => false
    })
  }

  test("lower gather") {
    val vocab = Dim("vocab", 10)
    val embed = Dim("embed", 4)
    val seq = Dim("seq", 3)
    val table = Ein.Param[Double]("E", List(vocab, embed), TensorData.zeros(List(vocab, embed)))
    val indices = TensorData.fromArray(List(seq), Array(0, 2, 1))
    val expr = Ein.Gather(table, indices, "vocab")

    val prog = Lower.lower(expr)

    assert(prog.ops.exists {
      case LowOp.Gather(_, _, _, _) => true
      case _ => false
    })
    assertEquals(prog.outputs.head.shape, List(3, 4))
  }

  test("lower reshape") {
    val d = Dim("d", 6)
    val x = Ein.Input[Double]("x", List(d))
    val target = List(Dim("r", 2), Dim("c", 3))
    val expr = Ein.Reshape(x, target)

    val prog = Lower.lower(expr)

    assert(prog.ops.exists {
      case LowOp.Reshape(_, _, _) => true
      case _ => false
    })
    assertEquals(prog.outputs.head.shape, List(2, 3))
  }

  test("lower all activations") {
    val d = Dim("d", 3)
    val x = Ein.Input[Double]("x", List(d))

    for act <- List(Activation.ReLU, Activation.Sigmoid, Activation.Tanh, Activation.GELU, Activation.Swish) do
      val expr = Ein.Activate(act, x)
      val prog = Lower.lower(expr)
      assert(prog.ops.exists {
        case LowOp.Activate(_, _, _) => true
        case _ => false
      }, s"Failed for activation $act")
  }

class CGenSuite extends munit.FunSuite:

  test("generate C code for simple MLP") {
    val inp = Dim("inp", 2)
    val hid = Dim("hid", 3)
    val out = Dim("out", 1)

    val w1 = Ein.Param[Double]("W1", List(hid, inp), TensorData.zeros(List(hid, inp)))
    val b1 = Ein.Param[Double]("b1", List(hid), TensorData.zeros(List(hid)))
    val w2 = Ein.Param[Double]("W2", List(out, hid), TensorData.zeros(List(out, hid)))
    val b2 = Ein.Param[Double]("b2", List(out), TensorData.zeros(List(out)))
    val x = Ein.Input[Double]("x", List(inp))

    val h = Ein.Activate(Activation.ReLU, (w1 * x) + b1)
    val output = (w2 * h) + b2

    val prog = Lower.lower(output)
    val code = CGen.generate(prog)

    assert(code.contains("#include <math.h>"), "Should include math.h")
    assert(code.contains("void forward(void)"), "Should have forward function")
    assert(code.contains("void init(void)"), "Should have init function")
    assert(code.contains("int main(void)"), "Should have main function")
    assert(code.contains("MatMul"), "Should have matmul op")
    assert(code.contains("v > 0 ? v : 0"), "Should have ReLU activation")
  }

  test("generate C code for block network with sigmoid") {
    val inp2 = Dim("inp", 2)
    val input = Ein.Input[Double]("x", List(inp2))
    val net = (Block.dense[Double](4, Activation.Sigmoid) >> Block.linear[Double](1))
      .materialize(input)

    val prog = Lower.lower(net)
    val code = CGen.generate(prog)

    assert(code.contains("1.0 / (1.0 + exp(-v))"), "Should have Sigmoid")
    assert(code.contains("void forward(void)"))
  }

  test("generated C code has all activations") {
    val d = Dim("d", 3)
    val x = Ein.Input[Double]("x", List(d))

    for act <- List(Activation.ReLU, Activation.Sigmoid, Activation.Tanh, Activation.GELU, Activation.Swish) do
      val expr = Ein.Activate(act, x)
      val prog = Lower.lower(expr)
      val code = CGen.generate(prog)

      act match
        case Activation.ReLU    => assert(code.contains("v > 0 ? v : 0"), s"Missing ReLU in C")
        case Activation.Sigmoid => assert(code.contains("1.0 / (1.0 + exp(-v))"), s"Missing Sigmoid in C")
        case Activation.Tanh    => assert(code.contains("tanh(v)"), s"Missing Tanh in C")
        case Activation.GELU    => assert(code.contains("0.5"), s"Missing GELU in C")
        case Activation.Swish   => assert(code.contains("exp(-v)"), s"Missing Swish in C")
  }

  test("C code for softmax") {
    val d = Dim("d", 4)
    val x = Ein.Input[Double]("x", List(d))
    val expr = Ein.Softmax(x, "d")
    val prog = Lower.lower(expr)
    val code = CGen.generate(prog)

    assert(code.contains("softmax") || code.contains("exp"), "Should have softmax computation")
  }

  test("C code for gather") {
    val vocab = Dim("vocab", 10)
    val embed = Dim("embed", 4)
    val seq = Dim("seq", 3)
    val table = Ein.Param[Double]("E", List(vocab, embed), TensorData.zeros(List(vocab, embed)))
    val indices = TensorData.fromArray(List(seq), Array(0, 2, 1))
    val expr = Ein.Gather(table, indices, "vocab")
    val prog = Lower.lower(expr)
    val code = CGen.generate(prog)

    assert(code.contains("Gather") || code.contains("gather"), "Should have gather op")
  }
