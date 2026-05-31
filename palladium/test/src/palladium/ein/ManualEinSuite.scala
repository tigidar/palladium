package palladium.ein

import palladium.ein.EinDsl.*
import palladium.ein.EinDsl.syntax.*

class ManualEinSuite extends munit.FunSuite:

  val eps = 1e-3

  test("minimal neural network using Ein (raw constructors)") {
    val inp = Dim("inp", 2)
    val out = Dim("out", 1)

    val x = Ein.Input[Double]("x", List(inp))

    val w = Ein.Param[Double](
      "w",
      List(out, inp),
      TensorData.fromArray(List(out, inp), Array(-3.0, 1.0))
    )
    val b = Ein.Param[Double](
      "b",
      List(out),
      TensorData.fromArray(List(out), Array(6.8813735870195432))
    )

    val n = w * x + b
    val o = Ein.Activate(Activation.Tanh, n)

    val feed = Map("x" -> TensorData.fromArray(List(inp), Array(2.0, 0.0)))

    val result = EinEval.eval(o, feed)
    assertEqualsDouble(result.data(0), 0.707, eps)
  }

  test("minimal neural network using EinDsl") {
    val inp = 2.dim
    val out = 1.dim

    val x = input[Double](inp)
    val w = weight(inp → out, -3.0, 1.0)
    val b = bias(out, 6.8813735870195432)

    val o = tanh(w * x + b)

    val result = o.eval(x.data(2.0, 0.0))
    assertEqualsDouble(result.data(0), 0.707, eps)
  }

  test("neural network with random initialization") {
    given rng: java.util.Random = java.util.Random(42)

    val inp = 2.dim
    val out = 1.dim

    val x = input[Double](inp)
    val w = weight(inp → out, uniform(-1, 1))
    val b = bias(out, zeros)

    val o = tanh(w * x + b)

    val result = o.eval(x.data(2.0, 0.0))
    // output is tanh of some random linear combination — just check it's in [-1, 1]
    assert(result.data(0) >= -1.0 && result.data(0) <= 1.0)
  }

  // Karpathy's MLP(3, [4, 4, 1]) — low level

  test("MLP 3 → 4 → 4 → 1 (low level)") {
    given rng: java.util.Random = java.util.Random(42)

    val inp = 3.dim
    val h1 = 4.dim
    val h2 = 4.dim
    val out = 1.dim

    val x = input[Double](inp)

    // layer 0: 3 → 4
    val w0 = weight(inp → h1, uniform(-1, 1))
    val b0 = bias(h1, zeros)
    val l0 = tanh(w0 * x + b0)

    // layer 1: 4 → 4
    val w1 = weight(h1 → h2, uniform(-1, 1))
    val b1 = bias(h2, zeros)
    val l1 = tanh(w1 * l0 + b1)

    // layer 2: 4 → 1
    val w2 = weight(h2 → out, uniform(-1, 1))
    val b2 = bias(out, zeros)
    val l2 = tanh(w2 * l1 + b2)

    val result = l2.eval(x.data(2.0, 3.0, -1.0))

    // tanh output is always in [-1, 1]
    assert(result.data(0) >= -1.0 && result.data(0) <= 1.0)
    assertEquals(result.dims, List(out))
  }

  // Same network — high level (Block DSL)

  test("MLP 3 → 4 → 4 → 1 (Block DSL)") {
    import Block.*
    given rng: java.util.Random = java.util.Random(42)

    val inp = 3.dim
    val x = input[Double](inp)

    val net =
      x
        >> dense(4, Activation.Tanh, uniform(-1, 1))
        >> dense(4, Activation.Tanh, uniform(-1, 1))
        >> dense(1, Activation.Tanh, uniform(-1, 1))

    val result = net.materialize.eval(x.data(2.0, 3.0, -1.0))

    assert(result.data(0) >= -1.0 && result.data(0) <= 1.0)
    assert(result.data(0) != 0.0, "should not be zero with random init")
    assertEquals(result.dims.size, 1)
  }

  // Multiple samples — loop over inputs (no batching needed)

  test("MLP on multiple samples with MSE loss") {
    import Block.*
    given rng: java.util.Random = java.util.Random(42)

    val inp = 3.dim
    val out = 1.dim
    val x = input[Double](inp)
    val target = input[Double](out)

    val neuralNetwork =
      x
        >> dense(4, Activation.Tanh, uniform(-1, 1))
        >> dense(4, Activation.Tanh, uniform(-1, 1))
        >> dense(1, Activation.Tanh, uniform(-1, 1))

    val expr = neuralNetwork.materialize

    // MSE loss: (pred - target)² summed over output dim
    val diff = expr - target
    val loss = diff.square.sum

    println(loss)

    val xs = Vector(
      Vector(2.0, 3.0, -1.0),
      Vector(3.0, -1.0, 0.5),
      Vector(0.5, 1.0, 1.0),
      Vector(1.0, 1.0, -1.0)
    )
    val ys = Vector(1.0, -1.0, -1.0, 1.0)

    // forward pass: predictions
    val yPred = xs.map(sample => expr.eval(x.data(sample*)))
    yPred.foreach { r =>
      assert(r.data(0) >= -1.0 && r.data(0) <= 1.0)
    }

    // loss for each sample
    val losses = xs.zip(ys).map { (sample, y) =>
      loss.eval(x.data(sample*), target.data(y))
    }

    println(losses)

    // each loss is (pred - target)² ≥ 0
    losses.foreach { l =>
      assert(l.data(0) >= 0.0, s"loss should be non-negative, got ${l.data(0)}")
    }

    // total loss = mean of individual losses
    val totalLoss = losses.map(_.data(0)).sum / losses.size
    assert(totalLoss > 0.0, "total loss should be > 0 before training")
  }

  // Training loop — compile + SGD

  test("train MLP with SGD until convergence") {
    import Block.*
    given rng: java.util.Random = java.util.Random(42)

    val inp = 3.dim
    val out = 1.dim
    val x = input[Double](inp)
    val target = input[Double](out)

    val pred =
      x
        >> dense(4, Activation.Tanh, uniform(-1, 1))
        >> dense(4, Activation.Tanh, uniform(-1, 1))
        >> dense(1, Activation.Tanh, uniform(-1, 1))

    val expr = pred.materialize
    val loss = (expr - target).square.sum

    // compile to flat-array loops — pure Scala, no BLAS
    val mlp = EinCompile.compile(loss)

    val xs = Array(
      Array(2.0, 3.0, -1.0),
      Array(3.0, -1.0, 0.5),
      Array(0.5, 1.0, 1.0),
      Array(1.0, 1.0, -1.0)
    )
    val ys = Array(1.0, -1.0, -1.0, 1.0)

    val result = mlp.train(xs, ys, lr = 0.1, epochs = 100, minLoss = 0.01)

    assert(result.converged, s"expected convergence, got loss ${result.finalLoss} after ${result.epochsRun} epochs")

    // verify predictions match targets
    for i <- xs.indices do
      val prediction = mlp.forward(xs(i))
      assertEqualsDouble(prediction(0), ys(i), 0.15)
  }
