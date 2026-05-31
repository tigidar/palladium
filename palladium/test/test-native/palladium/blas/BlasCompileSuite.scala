package palladium.blas

import palladium.ein.*
import palladium.ein.EinDsl.*
import palladium.ein.EinDsl.syntax.*
import palladium.ein.Block.*

class BlasCompileSuite extends munit.FunSuite:

  val eps = 1e-6

  test("BlasCompile forward matches EinCompile forward") {
    given rng: java.util.Random = java.util.Random(42)

    val inp = 3.dim
    val out = 1.dim
    val x = input[Double](inp)
    val target = input[Double](out)

    val pred = x >> dense(4, Activation.Tanh, uniform(-1, 1)) >> dense(1, Activation.Tanh, uniform(-1, 1))
    val expr = pred.materialize
    val loss = (expr - target).square.sum

    val jvmMlp = EinCompile.compile(loss)
    val blasMlp = BlasCompile.compile(loss)

    val xs = Array(Array(2.0, 3.0, -1.0), Array(0.5, 1.0, 1.0))

    for sample <- xs do
      val jvmResult = jvmMlp.forward(sample)
      val blasResult = blasMlp.forward(sample)
      assertEqualsDouble(blasResult(0), jvmResult(0), eps)
  }

  test("BlasCompile training converges") {
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

    val loss = (pred.materialize - target).square.sum
    val mlp = BlasCompile.compile(loss)

    val xs = Array(
      Array(2.0, 3.0, -1.0),
      Array(3.0, -1.0, 0.5),
      Array(0.5, 1.0, 1.0),
      Array(1.0, 1.0, -1.0)
    )
    val ys = Array(1.0, -1.0, -1.0, 1.0)

    val result = mlp.train(xs, ys, lr = 0.1, epochs = 100, minLoss = 0.01)
    assert(result.converged, s"expected convergence, got loss ${result.finalLoss} after ${result.epochsRun} epochs")

    // verify predictions
    for i <- xs.indices do
      val prediction = mlp.forward(xs(i))
      assertEqualsDouble(prediction(0), ys(i), 0.15)
  }

  test("BlasCompile and EinCompile converge to same loss") {
    val loss1 = {
      given rng: java.util.Random = java.util.Random(42)
      val inp = 3.dim; val out = 1.dim
      val x = input[Double](inp); val target = input[Double](out)
      val pred = x >> dense(4, Activation.Tanh, uniform(-1, 1)) >> dense(1, Activation.Tanh, uniform(-1, 1))
      (pred.materialize - target).square.sum
    }
    val loss2 = {
      given rng: java.util.Random = java.util.Random(42)
      val inp = 3.dim; val out = 1.dim
      val x = input[Double](inp); val target = input[Double](out)
      val pred = x >> dense(4, Activation.Tanh, uniform(-1, 1)) >> dense(1, Activation.Tanh, uniform(-1, 1))
      (pred.materialize - target).square.sum
    }

    val jvmMlp = EinCompile.compile(loss1)
    val blasMlp = BlasCompile.compile(loss2)

    val xs = Array(Array(2.0, 3.0, -1.0), Array(3.0, -1.0, 0.5))
    val ys = Array(1.0, -1.0)

    val jvmResult = jvmMlp.train(xs, ys, lr = 0.1, epochs = 50)
    val blasResult = blasMlp.train(xs, ys, lr = 0.1, epochs = 50)

    assertEqualsDouble(blasResult.finalLoss, jvmResult.finalLoss, 1e-10)
  }
