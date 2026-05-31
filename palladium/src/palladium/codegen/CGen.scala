package palladium.codegen

object CGen:

  def generate(prog: LowProgram): String =
    val sb = new StringBuilder
    val cType = prog.numType match
      case NumType.F64 => "double"
      case NumType.F32 => "float"

    // Header
    sb.append("#include <stdio.h>\n")
    sb.append("#include <stdlib.h>\n")
    sb.append("#include <string.h>\n")
    sb.append("#include <math.h>\n\n")

    // Buffer declarations
    val allBuffers = collectAllBuffers(prog)
    for (ref, label) <- allBuffers do
      sb.append(s"static $cType ${label}[${ref.totalSize}];\n")
    sb.append("\n")

    // init()
    sb.append("void init(void) {\n")
    for (ref, label) <- allBuffers do
      sb.append(s"  memset($label, 0, sizeof($label));\n")
    sb.append("}\n\n")

    // forward()
    sb.append("void forward(void) {\n")
    for op <- prog.ops do
      sb.append(emitOp(op, cType))
    sb.append("}\n\n")

    // main()
    sb.append("int main(void) {\n")
    sb.append("  init();\n\n")

    // Read inputs from stdin
    for (ref, name) <- prog.inputs do
      val label = bufLabel(ref)
      sb.append(s"  /* read input: $name */\n")
      sb.append(s"  for (int i = 0; i < ${ref.totalSize}; i++) {\n")
      if cType == "float" then
        sb.append(s"""    scanf("%f", &${label}[i]);\n""")
      else
        sb.append(s"""    scanf("%lf", &${label}[i]);\n""")
      sb.append("  }\n")
    sb.append("\n")

    // Read params from stdin
    for (ref, name) <- prog.params do
      val label = bufLabel(ref)
      sb.append(s"  /* read param: $name */\n")
      sb.append(s"  for (int i = 0; i < ${ref.totalSize}; i++) {\n")
      if cType == "float" then
        sb.append(s"""    scanf("%f", &${label}[i]);\n""")
      else
        sb.append(s"""    scanf("%lf", &${label}[i]);\n""")
      sb.append("  }\n")
    sb.append("\n")

    sb.append("  forward();\n\n")

    // Write outputs to stdout
    for outRef <- prog.outputs do
      val label = bufLabel(outRef)
      sb.append(s"  for (int i = 0; i < ${outRef.totalSize}; i++) {\n")
      if cType == "float" then
        sb.append(s"""    printf("%f\\n", ${label}[i]);\n""")
      else
        sb.append(s"""    printf("%.17g\\n", ${label}[i]);\n""")
      sb.append("  }\n")
    sb.append("\n")

    sb.append("  return 0;\n")
    sb.append("}\n")
    sb.toString

  private def bufLabel(ref: TensorRef): String = s"buf_${ref.id}"

  private def collectAllBuffers(prog: LowProgram): Vector[(TensorRef, String)] =
    val paramBufs = prog.params.map((ref, name) => (ref, bufLabel(ref)))
    val inputBufs = prog.inputs.map((ref, name) => (ref, bufLabel(ref)))
    val tempBufs = prog.tempBuffers.map(ref => (ref, bufLabel(ref)))
    paramBufs ++ inputBufs ++ tempBufs

  private def emitOp(op: LowOp, cType: String): String =
    val sb = new StringBuilder
    op match
      case LowOp.MatMul(out, left, right, outShape, leftFreeCount, rightFreeCount, contractSize) =>
        val o = bufLabel(out)
        val l = bufLabel(left)
        val r = bufLabel(right)
        val leftFreeSize = if leftFreeCount == 0 then 1 else left.totalSize / contractSize
        val rightFreeSize = if rightFreeCount == 0 then 1 else right.totalSize / contractSize
        sb.append(s"  /* MatMul: ${o} = ${l} x ${r} */\n")
        sb.append(s"  memset($o, 0, sizeof($cType) * ${out.totalSize});\n")
        sb.append(s"  for (int i = 0; i < $leftFreeSize; i++) {\n")
        sb.append(s"    for (int j = 0; j < $rightFreeSize; j++) {\n")
        sb.append(s"      $cType sum = 0;\n")
        sb.append(s"      for (int k = 0; k < $contractSize; k++) {\n")
        sb.append(s"        sum += ${l}[i * $contractSize + k] * ${r}[k * $rightFreeSize + j];\n")
        sb.append(s"      }\n")
        sb.append(s"      ${o}[i * $rightFreeSize + j] = sum;\n")
        sb.append(s"    }\n")
        sb.append(s"  }\n")

      case LowOp.ElemBinary(out, left, right, bop) =>
        val o = bufLabel(out)
        val l = bufLabel(left)
        val r = bufLabel(right)
        val opStr = bop match
          case BinaryOp.Add => "+"
          case BinaryOp.Sub => "-"
          case BinaryOp.Mul => "*"
        val leftSize = left.totalSize
        val rightSize = right.totalSize
        val outSize = out.totalSize
        sb.append(s"  /* ElemBinary: ${o} = ${l} $opStr ${r} */\n")
        sb.append(s"  for (int i = 0; i < $outSize; i++) {\n")
        sb.append(s"    ${o}[i] = ${l}[i % $leftSize] $opStr ${r}[i % $rightSize];\n")
        sb.append(s"  }\n")

      case LowOp.Activate(out, input, f) =>
        val o = bufLabel(out)
        val x = bufLabel(input)
        val size = out.totalSize
        sb.append(s"  /* Activate: ${activationName(f)} */\n")
        sb.append(s"  for (int i = 0; i < $size; i++) {\n")
        sb.append(s"    $cType v = ${x}[i];\n")
        sb.append(s"    ${o}[i] = ${activationExpr(f, "v")};\n")
        sb.append(s"  }\n")

      case LowOp.ActivateDeriv(out, input, f) =>
        val o = bufLabel(out)
        val x = bufLabel(input)
        val size = out.totalSize
        sb.append(s"  /* ActivateDeriv: ${activationName(f)} */\n")
        sb.append(s"  for (int i = 0; i < $size; i++) {\n")
        sb.append(s"    $cType v = ${x}[i];\n")
        sb.append(s"    ${o}[i] = ${activationDerivExpr(f, "v")};\n")
        sb.append(s"  }\n")

      case LowOp.ReduceSum(out, input, keepDims, reduceDims) =>
        val o = bufLabel(out)
        val x = bufLabel(input)
        val outSize = out.totalSize
        val reduceSize = if reduceDims.isEmpty then 1 else reduceDims.product
        sb.append(s"  /* ReduceSum */\n")
        sb.append(s"  memset($o, 0, sizeof($cType) * $outSize);\n")
        sb.append(s"  for (int i = 0; i < $outSize; i++) {\n")
        sb.append(s"    $cType sum = 0;\n")
        sb.append(s"    for (int j = 0; j < $reduceSize; j++) {\n")
        sb.append(s"      sum += ${x}[i * $reduceSize + j];\n")
        sb.append(s"    }\n")
        sb.append(s"    ${o}[i] = sum;\n")
        sb.append(s"  }\n")

      case LowOp.Broadcast(out, input, targetShape) =>
        val o = bufLabel(out)
        val x = bufLabel(input)
        val outSize = out.totalSize
        val inSize = input.totalSize
        sb.append(s"  /* Broadcast */\n")
        sb.append(s"  for (int i = 0; i < $outSize; i++) {\n")
        sb.append(s"    ${o}[i] = ${x}[i % $inSize];\n")
        sb.append(s"  }\n")

      case LowOp.Copy(out, input) =>
        val o = bufLabel(out)
        val x = bufLabel(input)
        val size = out.totalSize.min(input.totalSize)
        sb.append(s"  /* Copy */\n")
        sb.append(s"  memcpy($o, $x, sizeof($cType) * $size);\n")

      case LowOp.FillOnes(out) =>
        val o = bufLabel(out)
        val size = out.totalSize
        sb.append(s"  /* FillOnes */\n")
        sb.append(s"  for (int i = 0; i < $size; i++) ${o}[i] = 1.0;\n")

      case LowOp.FillZeros(out) =>
        val o = bufLabel(out)
        sb.append(s"  /* FillZeros */\n")
        sb.append(s"  memset($o, 0, sizeof($cType) * ${out.totalSize});\n")

      case LowOp.Softmax(out, input, overDimSize) =>
        val o = bufLabel(out)
        val x = bufLabel(input)
        val totalSize = out.totalSize
        val batchSize = totalSize / overDimSize
        sb.append(s"  /* Softmax */\n")
        sb.append(s"  for (int b = 0; b < $batchSize; b++) {\n")
        sb.append(s"    int base = b * $overDimSize;\n")
        sb.append(s"    $cType maxv = ${x}[base];\n")
        sb.append(s"    for (int i = 1; i < $overDimSize; i++) {\n")
        sb.append(s"      if (${x}[base + i] > maxv) maxv = ${x}[base + i];\n")
        sb.append(s"    }\n")
        sb.append(s"    $cType sum = 0;\n")
        sb.append(s"    for (int i = 0; i < $overDimSize; i++) {\n")
        sb.append(s"      ${o}[base + i] = exp(${x}[base + i] - maxv);\n")
        sb.append(s"      sum += ${o}[base + i];\n")
        sb.append(s"    }\n")
        sb.append(s"    for (int i = 0; i < $overDimSize; i++) {\n")
        sb.append(s"      ${o}[base + i] /= sum;\n")
        sb.append(s"    }\n")
        sb.append(s"  }\n")

      case LowOp.LogSoftmax(out, input, overDimSize) =>
        val o = bufLabel(out)
        val x = bufLabel(input)
        val totalSize = out.totalSize
        val batchSize = totalSize / overDimSize
        sb.append(s"  /* LogSoftmax */\n")
        sb.append(s"  for (int b = 0; b < $batchSize; b++) {\n")
        sb.append(s"    int base = b * $overDimSize;\n")
        sb.append(s"    $cType maxv = ${x}[base];\n")
        sb.append(s"    for (int i = 1; i < $overDimSize; i++) {\n")
        sb.append(s"      if (${x}[base + i] > maxv) maxv = ${x}[base + i];\n")
        sb.append(s"    }\n")
        sb.append(s"    $cType logsum = 0;\n")
        sb.append(s"    for (int i = 0; i < $overDimSize; i++) {\n")
        sb.append(s"      logsum += exp(${x}[base + i] - maxv);\n")
        sb.append(s"    }\n")
        sb.append(s"    logsum = log(logsum);\n")
        sb.append(s"    for (int i = 0; i < $overDimSize; i++) {\n")
        sb.append(s"      ${o}[base + i] = (${x}[base + i] - maxv) - logsum;\n")
        sb.append(s"    }\n")
        sb.append(s"  }\n")

      case LowOp.LayerNorm(out, input, scale, bias, normSize, eps) =>
        val o = bufLabel(out)
        val x = bufLabel(input)
        val g = bufLabel(scale)
        val b = bufLabel(bias)
        val totalSize = out.totalSize
        val batchSize = totalSize / normSize
        sb.append(s"  /* LayerNorm (eps=$eps) */\n")
        sb.append(s"  for (int b = 0; b < $batchSize; b++) {\n")
        sb.append(s"    int base = b * $normSize;\n")
        sb.append(s"    $cType mean = 0;\n")
        sb.append(s"    for (int i = 0; i < $normSize; i++) mean += ${x}[base + i];\n")
        sb.append(s"    mean /= $normSize;\n")
        sb.append(s"    $cType var_ = 0;\n")
        sb.append(s"    for (int i = 0; i < $normSize; i++) {\n")
        sb.append(s"      $cType d = ${x}[base + i] - mean;\n")
        sb.append(s"      var_ += d * d;\n")
        sb.append(s"    }\n")
        sb.append(s"    var_ /= $normSize;\n")
        sb.append(s"    $cType inv_std = 1.0 / sqrt(var_ + $eps);\n")
        sb.append(s"    for (int i = 0; i < $normSize; i++) {\n")
        sb.append(s"      ${o}[base + i] = (${x}[base + i] - mean) * inv_std * ${g}[i] + ${b}[i];\n")
        sb.append(s"    }\n")
        sb.append(s"  }\n")

      case LowOp.Reshape(out, input, targetShape) =>
        val o = bufLabel(out)
        val x = bufLabel(input)
        val size = out.totalSize.min(input.totalSize)
        sb.append(s"  /* Reshape */\n")
        sb.append(s"  memcpy($o, $x, sizeof($cType) * $size);\n")

      case LowOp.Gather(out, table, indicesId, indicesSize) =>
        val o = bufLabel(out)
        val t = bufLabel(table)
        val embDim = table.totalSize / (if table.shape.nonEmpty then table.shape.head else 1)
        sb.append(s"  /* Gather (embedding lookup) */\n")
        sb.append(s"  for (int i = 0; i < $indicesSize; i++) {\n")
        sb.append(s"    int idx = indices_data[i];\n")
        sb.append(s"    for (int j = 0; j < $embDim; j++) {\n")
        sb.append(s"      ${o}[i * $embDim + j] = ${t}[idx * $embDim + j];\n")
        sb.append(s"    }\n")
        sb.append(s"  }\n")

      case LowOp.Scatter(out, src, indicesId, indicesSize) =>
        val o = bufLabel(out)
        val s = bufLabel(src)
        val embDim = if src.shape.size > 1 then src.shape.last else 1
        sb.append(s"  /* Scatter */\n")
        sb.append(s"  memset($o, 0, sizeof($cType) * ${out.totalSize});\n")
        sb.append(s"  for (int i = 0; i < $indicesSize; i++) {\n")
        sb.append(s"    int idx = indices_data[i];\n")
        sb.append(s"    for (int j = 0; j < $embDim; j++) {\n")
        sb.append(s"      ${o}[idx * $embDim + j] += ${s}[i * $embDim + j];\n")
        sb.append(s"    }\n")
        sb.append(s"  }\n")

    sb.toString

  private def activationName(f: ActivationType): String = f match
    case ActivationType.ReLU    => "ReLU"
    case ActivationType.Sigmoid => "Sigmoid"
    case ActivationType.Tanh    => "Tanh"
    case ActivationType.GELU    => "GELU"
    case ActivationType.Swish   => "Swish"

  private def activationExpr(f: ActivationType, v: String): String = f match
    case ActivationType.ReLU    => s"($v > 0 ? $v : 0)"
    case ActivationType.Sigmoid => s"(1.0 / (1.0 + exp(-$v)))"
    case ActivationType.Tanh    => s"tanh($v)"
    case ActivationType.GELU    =>
      s"(0.5 * $v * (1.0 + tanh(sqrt(2.0 / 3.14159265358979323846) * ($v + 0.044715 * $v * $v * $v))))"
    case ActivationType.Swish   =>
      s"($v / (1.0 + exp(-$v)))"

  private def activationDerivExpr(f: ActivationType, v: String): String = f match
    case ActivationType.ReLU    => s"($v > 0 ? 1.0 : 0.0)"
    case ActivationType.Sigmoid =>
      s"({ double s_ = 1.0 / (1.0 + exp(-$v)); s_ * (1.0 - s_); })"
    case ActivationType.Tanh    =>
      s"({ double t_ = tanh($v); 1.0 - t_ * t_; })"
    case ActivationType.GELU    =>
      val sqrtCoeff = "sqrt(2.0 / 3.14159265358979323846)"
      s"""({ double u_ = $sqrtCoeff * ($v + 0.044715 * $v * $v * $v); \\
      double t_ = tanh(u_); \\
      double du_ = $sqrtCoeff * (1.0 + 3.0 * 0.044715 * $v * $v); \\
      0.5 * (1.0 + t_) + 0.5 * $v * (1.0 - t_ * t_) * du_; })"""
    case ActivationType.Swish   =>
      s"({ double s_ = 1.0 / (1.0 + exp(-$v)); s_ * (1.0 + $v * (1.0 - s_)); })"
