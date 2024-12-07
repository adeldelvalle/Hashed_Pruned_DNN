
import torch
from cupy_kernel import cupyKernel
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
kernel = '''


extern "C"
__global__ void fingerprint(const float* src, const int k, const int L, unsigned long long* fp) {
    // We assume k ≤ 256 for demonstration.
    // Each thread corresponds to one bit position if threadIdx.x < k.
    // Otherwise, it contributes no bit.

    int global_sample_id = blockIdx.x; // N dimension in grid
    int table_id = blockIdx.y;         // L dimension in grid
    int tid = threadIdx.x;

    // Compute data index
    int offset = (k * L * global_sample_id) + (k * table_id + tid);

    // Determine if this thread sets a bit (if tid < k)
    bool set_bit = (tid < k && src[offset] > 0.0f);

    // We'll store bits in four 64-bit segments to cover up to 256 bits.
    // tid goes from 0 to 255, we can identify segment and bit_pos:
    int segment = tid / 64;       // which 64-bit block (0 to 3)
    int bit_pos = tid % 64;       // which bit within that 64-bit block

    // Each thread starts with zeroed out segments
    unsigned long long val[4] = {0ULL, 0ULL, 0ULL, 0ULL};

    if (set_bit) {
        val[segment] = 1ULL << bit_pos;
    }

    // Now we have 8 warps (256/32 = 8). Each warp can reduce within itself using __shfl_down_sync.
    // Warp-level reduction for each segment.
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        unsigned long long x = val[s];
        // Reduce within a warp (32 threads)
        for (int offset = 16; offset > 0; offset /= 2) {
            x |= __shfl_down_sync(0xFFFFFFFF, x, offset, 32);
        }
        // After this, only lane 0 in each warp holds the reduced result for that segment.
        val[s] = x;
    }

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // We need shared memory to combine results from all warps.
    __shared__ unsigned long long warp_results[4 * 8]; // 4 segments * 8 warps = 32 ull
    if (lane_id == 0) {
        // Write each segment's result for this warp
        warp_results[warp_id * 4 + 0] = val[0];
        warp_results[warp_id * 4 + 1] = val[1];
        warp_results[warp_id * 4 + 2] = val[2];
        warp_results[warp_id * 4 + 3] = val[3];
    }
    __syncthreads();

    // Now reduce across warps. We have 8 warps, each produced 4 segments.
    // We'll have one thread (e.g., threadIdx.x == 0) do the final reduction.
    if (tid == 0) {
        // Combine results from all 8 warps for each segment
        unsigned long long final_val[4] = {0ULL, 0ULL, 0ULL, 0ULL};
        for (int w = 0; w < 8; w++) {
            final_val[0] |= warp_results[w * 4 + 0];
            final_val[1] |= warp_results[w * 4 + 1];
            final_val[2] |= warp_results[w * 4 + 2];
            final_val[3] |= warp_results[w * 4 + 3];
        }

        // final_val now holds the combined 256-bit hash for (global_sample_id, table_id).

        // fp is assumed to be sized to hold these bits. If we store them continuously,
        // e.g., each entry in fp is 4 unsigned long long per item, we need a known layout.
        // For example:
        int fp_offset = L * global_sample_id + table_id;

        // Write out the first 64 bits (final_val[0]) to fp
        // If we want all 256 bits stored consecutively, we need fp dimensioning accordingly.
        // Suppose fp dimension is N*L*4 to store all 4 segments:
        int base = fp_offset * 4;
        fp[base + 0] = final_val[0];
        fp[base + 1] = final_val[1];
        fp[base + 2] = final_val[2];
        fp[base + 3] = final_val[3];
    }
}

'''

class SimHash:
    def __init__(self, d_, k_, L_, weights=None,seed_=8191):
        self.d = d_
        self.k = k_
        self.L = L_
        self.fp = cupyKernel(kernel, "fingerprint")


        if weights is None:
            self.rp = SimHash.generate(d_, k_, L_, seed_)
        else:
            self.rp = SimHash.generate_from_weight(weights)   

    # def generate_from_weight(weights):
    #     #print("generated from triplet weight")
    #     return weights.to(device)
    
    # def generate(d, k, L, seed):
    #     return torch.randn(d, k*L).to(device)
        
    def generate_from_weight(weights):
        #print("generated from triplet weight")
        matrix = weights
        positive = torch.gt(matrix, 0).int()
        negative = (matrix < 0.0).int()
        result = (positive - negative).float()
        #return result.cpu()
        return result.to(device)
    
    def generate(d, k, L, seed):
        #print("random generate hash table weight")
        rand_gen = np.random.RandomState(seed)
        matrix = rand_gen.randn(d, k*L)
        positive = np.greater_equal(matrix, 0.0)
        negative = np.less(matrix, 0.0)
        result = positive.astype(np.float32) - negative.astype(np.float32)
        #print("Shape of hash table:", torch.from_numpy(result).to(device).shape)
        return torch.from_numpy(result).to(device)

    # def generate_from_list(srp_list):
    #     matrices = [item.rp for item in srp_list]
    #     return torch.cat(matrices, dim=1)


    def hash(self, data, transpose=False):
        N, D = data.size()
        srp = torch.matmul(data.to(device), self.rp)
        #print("srp", srp)
        result = self.fingerprint(srp, N)
        #print("result", result)
        if transpose:
            result = torch.t(result) 
        return result

    # def hash(self, data, transpose=False):
    #     N, D = data.size()
    #     srp = torch.matmul(data, self.rp)
    #     positive = torch.gt(srp, 0).int()
    #     negative = (srp < 0.0).int()
    #     srp = (positive - negative).float()
    #     result = self.fingerprint(srp, N)
    #     if transpose:
    #         result = torch.t(result) 
    #     return result

    def fingerprint(self, srp, N):
        result = torch.zeros(N, self.L).long().to(device)
        self.fp(grid=(N,self.L,1),
                block=(32,1,1),
                args=[srp.data_ptr(), self.k, self.L, result.data_ptr()],
                strm=torch.cuda.current_stream().cuda_stream)
        return result.int()
