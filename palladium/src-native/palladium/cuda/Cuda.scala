package palladium.cuda

import scala.scalanative.unsafe.*
import scala.scalanative.unsigned.*

/** Minimal CUDA Runtime FFI bindings.
  *
  * We bind only what's needed for GPU buffer management:
  *   - cudaMalloc, cudaFree         — device memory
  *   - cudaMemcpy                   — host ↔ device transfer
  *   - cudaDeviceSynchronize        — sync for timing
  */
@link("cudart")
@extern
object Cuda:

  type CudaError = CInt

  /** Allocate memory on the GPU device. */
  def cudaMalloc(devPtr: Ptr[Ptr[Byte]], size: CSize): CudaError = extern

  /** Free GPU device memory. */
  def cudaFree(devPtr: Ptr[Byte]): CudaError = extern

  /** Copy data between host and device.
    * kind: 1 = HostToDevice, 2 = DeviceToHost, 3 = DeviceToDevice
    */
  def cudaMemcpy(dst: Ptr[Byte], src: Ptr[Byte], count: CSize, kind: CInt): CudaError = extern

  /** Block until all preceding CUDA calls complete. */
  def cudaDeviceSynchronize(): CudaError = extern

  /** Set zeros on device memory. */
  def cudaMemset(devPtr: Ptr[Byte], value: CInt, count: CSize): CudaError = extern

object CudaConstants:
  final val cudaMemcpyHostToDevice: CInt     = 1
  final val cudaMemcpyDeviceToHost: CInt     = 2
  final val cudaMemcpyDeviceToDevice: CInt   = 3
  final val cudaSuccess: CInt                = 0
