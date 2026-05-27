import cuda.tile as ct
import cupy

TILE_SIZE = 16

@ct.kernel
def block_jacobi_kernel(A, b, x, x_new):
  block_id0 = ct.bid(0)
  block_id1 = ct.bid(1)
  print(f"Processing block ({block_id0}, {block_id1})")
  # Load the block of A and corresponding entries of b and x
  A_block = ct.load(A, index=(block_id0, block_id1), shape=(TILE_SIZE, TILE_SIZE))
  b_block = ct.load(b, index=(block_id0,), shape=(TILE_SIZE,))
  x_block = ct.load(x, index=(block_id1,), shape=(TILE_SIZE,))
  
  # Compute the new values for this block
  x_new_block = b_block - A_block @ x_block
  # Store the results back to x_new
  ct.store(x_new, index=(block_id0,), tile=x_new_block)

  # Remove the diagonal contribution
  for i in range(TILE_SIZE):
    A_entry = ct.load(A, index=(block_id0 * TILE_SIZE + i, block_id0 * TILE_SIZE + i), shape=())
    x_new_entry = ct.load(x_new, index=(block_id0 * TILE_SIZE + i,), shape=())
    x_block_entry = ct.load(x, index=(block_id0 * TILE_SIZE + i,), shape=())
    x_new_entry += A_entry * x_block_entry
    x_new_entry /= A_entry
    ct.store(x_new, index=(block_id0 * TILE_SIZE + i,), tile=x_new_entry)
    # x_new_block[i] += A_block[i, i] * x_block[i]
    # x_new_block[i] /= A_block[i, i]

def block_jacobi(A: cupy.ndarray, b: cupy.ndarray, x: cupy.ndarray, x_new: cupy.ndarray):
  assert A.shape[0] == A.shape[1] == b.shape[0] == x.shape[0] == x_new.shape[0]
  grid = (ct.cdiv(A.shape[0], TILE_SIZE), 1, 1)
  ct.launch(cupy.cuda.get_current_stream(), grid, block_jacobi_kernel, (A, b, x, x_new))

def block_jacobi_cpu(A: cupy.ndarray, b: cupy.ndarray, x: cupy.ndarray, x_new: cupy.ndarray):
  n = A.shape[0]
  for i in range(n):
    sum_ax = 0.0
    for j in range(n):
      if j != i:
        sum_ax += A[i, j] * x[j]
    x_new[i] = (b[i] - sum_ax) / A[i, i]

if __name__ == "__main__":
  # Example usage
  size = 1024
  # a = cupy.random.rand(size).astype(cupy.float32)
  # b = cupy.random.rand(size).astype(cupy.float32)
  # result = cupy.empty_like(a)
  
  # vector_add(a, b, result)
  
  # # Verify the result
  # assert cupy.allclose(result, a + b)
  # print("Vector addition successful!")
  cupy.random.seed(0)
  A = cupy.random.rand(size, size).astype(cupy.float32)
  b = cupy.random.rand(size).astype(cupy.float32)
  x = cupy.zeros(size, dtype=cupy.float32)
  x_new = cupy.zeros_like(x)
  block_jacobi(A, b, x, x_new)

  x_new_cpu = cupy.zeros_like(x_new)
  block_jacobi_cpu(A, b, x, x_new_cpu)
  for i in range(size):
    if not cupy.isclose(x_new[i], x_new_cpu[i]):
      print(f"Mismatch at index {i}: GPU={x_new[i]}, CPU={x_new_cpu[i]}")
      break
  assert cupy.allclose(x_new, x_new_cpu)
  print(x_new)