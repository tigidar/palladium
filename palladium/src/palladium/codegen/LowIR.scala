package palladium.codegen

case class TensorRef(id: Int, shape: List[Int]):
  def totalSize: Int = if shape.isEmpty then 1 else shape.product

enum NumType:
  case F64, F32

enum BinaryOp:
  case Add, Sub, Mul

enum ActivationType:
  case ReLU, Sigmoid, Tanh, GELU, Swish

enum LowOp:
  case MatMul(out: TensorRef, left: TensorRef, right: TensorRef,
              outShape: List[Int], leftFreeCount: Int, rightFreeCount: Int, contractSize: Int)
  case ElemBinary(out: TensorRef, left: TensorRef, right: TensorRef, op: BinaryOp)
  case Activate(out: TensorRef, input: TensorRef, f: ActivationType)
  case ActivateDeriv(out: TensorRef, input: TensorRef, f: ActivationType)
  case ReduceSum(out: TensorRef, input: TensorRef, keepDims: List[Int], reduceDims: List[Int])
  case Broadcast(out: TensorRef, input: TensorRef, targetShape: List[Int])
  case Copy(out: TensorRef, input: TensorRef)
  case FillOnes(out: TensorRef)
  case FillZeros(out: TensorRef)
  case Softmax(out: TensorRef, input: TensorRef, overDimSize: Int)
  case LogSoftmax(out: TensorRef, input: TensorRef, overDimSize: Int)
  case LayerNorm(out: TensorRef, input: TensorRef, scale: TensorRef, bias: TensorRef,
                 normSize: Int, eps: Double)
  case Reshape(out: TensorRef, input: TensorRef, targetShape: List[Int])
  case Gather(out: TensorRef, table: TensorRef, indicesId: Int, indicesSize: Int)
  case Scatter(out: TensorRef, src: TensorRef, indicesId: Int, indicesSize: Int)

case class LowProgram(
  ops: Vector[LowOp],
  params: Vector[(TensorRef, String)],
  inputs: Vector[(TensorRef, String)],
  outputs: Vector[TensorRef],
  tempBuffers: Vector[TensorRef],
  numType: NumType = NumType.F64,
  indicesData: Map[Int, Array[Int]] = Map.empty
)
