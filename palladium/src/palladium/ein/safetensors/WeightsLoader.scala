package palladium.ein.safetensors

import palladium.NumberLike
import palladium.ein.{Dim, Ein, TensorData}
import no.virtual_architect.safetensors as ns
import scala.reflect.ClassTag

// Re-export upstream types so existing imports of `palladium.ein.safetensors.*`
// keep working unchanged.
type DType = ns.DType
val DType = ns.DType
type TensorMeta = ns.TensorMeta
val TensorMeta = ns.TensorMeta
type SafeTensorsHeader = ns.SafeTensorsHeader
val SafeTensorsHeader = ns.SafeTensorsHeader

object SafeTensors:

  /** Parse the 8-byte LE header-size prefix plus JSON header. */
  export ns.SafeTensors.parseHeader

  /** Read a tensor as TensorData[Float] with palladium named dims. */
  def readFloat(
      bytes: Array[Byte],
      header: SafeTensorsHeader,
      name: String,
      dimNames: List[String]
  ): TensorData[Float] =
    val (shape, data) = ns.SafeTensors.readFloat(bytes, header, name)
    require(dimNames.length == shape.length,
      s"dimNames length ${dimNames.length} != shape length ${shape.length} for tensor '$name'")
    val dims = dimNames.zip(shape).map((n, s) => Dim(n, s))
    TensorData.fromArray(dims, data)

  /** Read a tensor as TensorData[Double] with palladium named dims. */
  def readDouble(
      bytes: Array[Byte],
      header: SafeTensorsHeader,
      name: String,
      dimNames: List[String]
  ): TensorData[Double] =
    val (shape, data) = ns.SafeTensors.readDouble(bytes, header, name)
    require(dimNames.length == shape.length,
      s"dimNames length ${dimNames.length} != shape length ${shape.length} for tensor '$name'")
    val dims = dimNames.zip(shape).map((n, s) => Dim(n, s))
    TensorData.fromArray(dims, data)

  /** Load all tensors as a Map.
    *
    * @param dimMapping function from tensor name to dimension names for that tensor
    */
  def loadAll[A: NumberLike: ClassTag](
      bytes: Array[Byte],
      dimMapping: String => List[String]
  ): Map[String, TensorData[A]] =
    val header = ns.SafeTensors.parseHeader(bytes)
    val cls = summon[ClassTag[A]].runtimeClass
    header.tensors.map { (name, _) =>
      val dimNames = dimMapping(name)
      val td =
        if cls == classOf[Float] then
          readFloat(bytes, header, name, dimNames).asInstanceOf[TensorData[A]]
        else if cls == classOf[Double] then
          readDouble(bytes, header, name, dimNames).asInstanceOf[TensorData[A]]
        else
          throw IllegalArgumentException(s"Unsupported type: $cls")
      name -> td
    }

  /** Replace Ein.Param data in an expression tree with loaded weights. */
  def loadWeights[A: NumberLike: ClassTag](
      expr: Ein[A],
      weights: Map[String, TensorData[A]]
  ): Ein[A] =
    expr match
      case Ein.Param(id, dims, data) =>
        weights.get(id) match
          case Some(newData) =>
            require(dims.map(_.size) == newData.dims.map(_.size),
              s"Shape mismatch for '$id': expected ${dims.map(_.size)}, got ${newData.dims.map(_.size)}")
            Ein.Param(id, dims, newData)
          case None => expr
      case Ein.Contract(l, r) =>
        Ein.Contract(loadWeights(l, weights), loadWeights(r, weights))
      case Ein.ElemAdd(l, r) =>
        Ein.ElemAdd(loadWeights(l, weights), loadWeights(r, weights))
      case Ein.ElemSub(l, r) =>
        Ein.ElemSub(loadWeights(l, weights), loadWeights(r, weights))
      case Ein.ElemMul(l, r) =>
        Ein.ElemMul(loadWeights(l, weights), loadWeights(r, weights))
      case Ein.Activate(f, arg) =>
        Ein.Activate(f, loadWeights(arg, weights))
      case Ein.ActivateDeriv(f, arg) =>
        Ein.ActivateDeriv(f, loadWeights(arg, weights))
      case Ein.ReduceSum(arg, over) =>
        Ein.ReduceSum(loadWeights(arg, weights), over)
      case Ein.Broadcast(arg, td) =>
        Ein.Broadcast(loadWeights(arg, weights), td)
      case Ein.Transpose(arg, perm) =>
        Ein.Transpose(loadWeights(arg, weights), perm)
      case Ein.Softmax(arg, dim) =>
        Ein.Softmax(loadWeights(arg, weights), dim)
      case Ein.LogSoftmax(arg, dim) =>
        Ein.LogSoftmax(loadWeights(arg, weights), dim)
      case Ein.LayerNorm(arg, scale, bias, overDims, eps) =>
        Ein.LayerNorm(loadWeights(arg, weights), loadWeights(scale, weights),
          loadWeights(bias, weights), overDims, eps)
      case Ein.Reshape(arg, td) =>
        Ein.Reshape(loadWeights(arg, weights), td)
      case Ein.Slice(arg, dim, from, to) =>
        Ein.Slice(loadWeights(arg, weights), dim, from, to)
      case Ein.Gather(table, indices, dim) =>
        Ein.Gather(loadWeights(table, weights), indices, dim)
      case Ein.Scatter(src, indices, dim, tableDims) =>
        Ein.Scatter(loadWeights(src, weights), indices, dim, tableDims)
      case _ => expr // Input, Fill, Ones, Zeros — no children with params
