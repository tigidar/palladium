package palladium.ein

import palladium.ein.EinDsl.*
import sourcecode.Name

class ScopeSuite extends munit.FunSuite:

  // A reusable layer exactly like the README's: leaves named from their
  // bindings (w/b), discriminated by an ambient Scope taken from the call's
  // own binding name via sourcecode.Name.
  def linear(in: Dim, out: Dim)(x: Ein[Double])(using Scope, nm: Name, rng: java.util.Random): Ein[Double] =
    scoped(nm.value) {
      val w = weight(in -> out, xavier)
      val b = bias(out, zeros)
      (w * x) + b
    }

  given rng: java.util.Random = java.util.Random(0L)

  test("root scope leaves leaf ids unprefixed (backward compatible)") {
    val out = 3.dim
    val b = bias(out, 0.1, 0.2, 0.3)
    assertEquals(b.parameters.keySet, Set("b"))
  }

  test("linear captures its own binding name as the scope") {
    val in = 4.dim
    val out = 3.dim
    val x = input[Double](in)
    val q = linear(in, out)(x)
    assertEquals(q.parameters.keySet, Set("q.w", "q.b"))
  }

  test("Name capture still sees the binding through a wrapping call") {
    val in = 4.dim
    val out = 3.dim
    val x = input[Double](in)
    val h = gelu(linear(in, out)(x))
    assertEquals(h.parameters.keySet, Set("h.w", "h.b"))
  }

  test("distinct straight-line vals get distinct scopes") {
    val model = 8.dim
    val dk = 4.dim
    val x = input[Double](model)
    val q = linear(model, dk)(x)
    val k = linear(model, dk)(x)
    val v = linear(model, dk)(x)
    val all = (q.parameters ++ k.parameters ++ v.parameters).keySet
    assertEquals(all, Set("q.w", "q.b", "k.w", "k.b", "v.w", "v.b"))
  }

  test("scoped adds a per-iteration prefix in loops") {
    val in = 4.dim
    val out = 3.dim
    val x = input[Double](in)
    val heads = (0 until 2).map { h =>
      scoped(s"h$h") {
        val o = linear(in, out)(x)
        o
      }
    }.reduce(_ + _)
    assertEquals(
      heads.parameters.keySet,
      Set("h0.o.w", "h0.o.b", "h1.o.w", "h1.o.b")
    )
  }

  test("nested scopes compose with dots") {
    val in = 4.dim
    val out = 3.dim
    val x = input[Double](in)
    val block = scoped("layer0") {
      scoped("attn") {
        val proj = linear(in, out)(x)
        proj
      }
    }
    assertEquals(block.parameters.keySet, Set("layer0.attn.proj.w", "layer0.attn.proj.b"))
  }

  // --- Dim-typed overloads ---

  test("renameDim renames axes preserving size") {
    val seq = 4.dim
    val model = 8.dim
    val x = input[Double](seq, model)
    val kSeq = seq.size.dim
    val renamed = x.renameDim(seq -> kSeq)
    assertEquals(renamed.outputDims, List(Dim("kSeq", 4), model))
  }

  test("renameDim rejects a size change") {
    val seq = 4.dim
    val other = 5.dim
    val x = input[Double](seq)
    intercept[IllegalArgumentException](x.renameDim(seq -> other))
  }

  test("softmax accepts a Dim value") {
    val cls = 3.dim
    val x = input[Double](cls)
    x.softmax(cls) match
      case Ein.Softmax(_, over) => assertEquals(over, "cls")
      case other                => fail(s"expected Softmax, got $other")
  }

  test("sumOver accepts Dim values") {
    val r = 2.dim
    val c = 3.dim
    val x = input[Double](r, c)
    x.sumOver(c) match
      case Ein.ReduceSum(_, over) => assertEquals(over, List("c"))
      case other                  => fail(s"expected ReduceSum, got $other")
  }
