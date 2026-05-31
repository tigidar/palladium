package palladium.nn

import palladium.{NumberLike, Value}
import scala.compiletime.{constValue, erasedValue}

case class MLP[Shape <: Tuple] private (
    layers: Vector[Int],
    params: Map[String, Double]
):
  require(layers.length >= 2, s"MLP needs at least 2 layers (input + output), got ${layers.length}")

  val paramCount: Int = params.size

  val paramNames: Set[String] = params.keySet

  def forward(inputs: Vector[Value[Double]]): Vector[Value[Double]] =
    require(
      inputs.size == layers.head,
      s"Expected ${layers.head} inputs, got ${inputs.size}"
    )

    val layerPairs = layers.sliding(2).toVector.zipWithIndex

    layerPairs.foldLeft(inputs) { case (activations, (pair, layerIdx)) =>
      val fanIn = pair(0)
      val fanOut = pair(1)
      val isLastLayer = layerIdx == layerPairs.size - 1

      Vector.tabulate(fanOut) { j =>
        val weighted = (0 until fanIn).foldLeft(
          Value.Var(s"b${layerIdx}_$j", params(s"b${layerIdx}_$j")): Value[Double]
        ) { (acc, i) =>
          val wName = s"w${layerIdx}_${i}_$j"
          acc + Value.Var(wName, params(wName)) * activations(i)
        }
        if isLastLayer then weighted else weighted.tanh
      }
    }

  def withParams(newParams: Map[String, Double]): MLP[Shape] =
    copy(params = params ++ newParams)

object MLP:
  inline def init[Shape <: Tuple](seed: Long): MLP[Shape] =
    val layers = tupleToVector[Shape]
    initFromLayers[Shape](layers, seed)

  private inline def tupleToVector[T <: Tuple]: Vector[Int] = inline erasedValue[T] match
    case _: EmptyTuple => Vector.empty
    case _: (h *: t)   => constValue[h & Int] +: tupleToVector[t]

  /** Runtime factory: initialize from a Vector of layer sizes. */
  def initFromVector(layers: Vector[Int], seed: Long): MLP[EmptyTuple] =
    initFromLayers[EmptyTuple](layers, seed)

  /** Runtime factory: construct from existing parameters. */
  def fromVector(layers: Vector[Int], params: Map[String, Double]): MLP[EmptyTuple] =
    MLP[EmptyTuple](layers, params)

  private def initFromLayers[Shape <: Tuple](layers: Vector[Int], seed: Long): MLP[Shape] =
    val rng = java.util.Random(seed)
    val layerPairs = layers.sliding(2).toVector.zipWithIndex

    val params = layerPairs.flatMap { case (pair, layerIdx) =>
      val fanIn = pair(0)
      val fanOut = pair(1)
      val limit = math.sqrt(6.0 / (fanIn + fanOut))

      val weights = for
        i <- 0 until fanIn
        j <- 0 until fanOut
      yield s"w${layerIdx}_${i}_$j" -> (rng.nextDouble() * 2.0 * limit - limit)

      val biases = (0 until fanOut).map(j => s"b${layerIdx}_$j" -> 0.0)

      weights ++ biases
    }.toMap

    MLP[Shape](layers, params)
