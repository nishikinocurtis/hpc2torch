#include <cuda.h>
#include <stdio.h>

// consider 2D parallel: 1D for indices, 1D for data
template<typename T>
__global__ void gather1D(const T *data, const int32_t *indices, T *out, int64_t *dsize, int64_t *isize, int64_t pi_tail_dims, int total, int total_y, int axis = 0) {
    int thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    // indices made into total slices
    // scalar case: always indices[0]
    // 1D case: x := (rsize[0] / total) * thread_x .. min(rsize[0], (rsize[0] / total) * (thread_x + 1))
    // idv = indices[x]
    // 2D case: 
    // x := (rsize[0] / total) * thread_x .. min(rsize[0], (rsize[0] / total) * (thread_x + 1)), y := 0 .. rsize[1]
    // idv = indices[x * rsize[1] + y]

    int64_t pi_other_dims = axis == 0 ? pi_tail_dims * dsize[1] : pi_tail_dims;

    if (axis == 0) {
        int x_bound = min(isize[0], (isize[0] / total) * (thread_x + 1))
        #pragma unroll
        for (int x = isize[0] / total * thread_x, x < x_cound; ++x) {
            int idv = indices[x];
            int rp_bound = min(pi_other_dims, (pi_other_dims / total_y) / (thread_y + 1));
            #pragma unroll
            for (int rp = pi_other_dims / total_y * thread_y, rp < rp_bound; ++rp) {
                out[x * pi_other_dims + rp] = data[idv * pi_other_dims + rp];
            }
        }
    } else if (axis == 1) {
        // axis1, 1D:
        // for d0, d2, .. dr-1:, most possibly, only d0
        // how to ensure contigious access?
        // try not to access too many d0, so divide d0 only
        /*
            let idx0 := d0 / total_y * thread_y .. d0 / total_y * (thread_y + 1)
            for all other dims rp := 0 .. pi(d2, .. dr-1) - 1 (maybe only 0.)
            out[idx0, x, d2, ... dr-1] = data[idx0, idv, d2, ... dr-1]
            out[idx0 * pi_other + x * pi_2_other + rp] = data[idx0 * pi_other + idv * pi_2_other + rp]
        */
        int x_bound = min(isize[0], (isize[0] / total) * (thread_x + 1))
        #pragma unroll
        for (int x = isize[0] / total * thread_x, x < x_cound; ++x) {
            int idv = indices[x];
            int idx0_bound = min(dsize[0], dsize[0] / total_y * (thread_y + 1))
            #pragma unroll
            for (int idx0 = dsize[0] / total_Y * thread_y; idx0 < idx0_bound; ++idx0) {
                int pi_other = pi_other_dims * isize[0];
                int pi_data_other = pi_other_dims * dsize[1];
                int pi_2_other = pi_other_dims;
                #pragma unroll
                for (int rp = 0; rp < pi_2_other; ++rp) {
                    out[idx0 * pi_other + x * pi_2_other + rp] = data[idx0 * pi_data_other + idv * pi_2_other + rp];
                }
            }
        }
    }
    

    // axis0, 0D:
    // out[rp] = data[indices[0] * pi_other_dims + rp]

    // axis0, 1D:
    // for all other axis: d1, d2, .. dr-1
    // out[x, d1, .. dr-1] = data[idv, d1, .. dr-1]
    // divided into pic = dsize[1] * .. *dsize[r-1] / total_y pieces
    // for rp := pic * thread_y .. min(pi_other, pic * (thread_y + 1))
    // try to ensure contigious access:
    // don't cut earlier dims, cut total
    // let pi_other_dims = dsize[1] * .. *dsize[r-1]
    // out[x * pi_other_dims + rp] = data[idv * pi_other_dims + rp]

    // axis0, 2D:
    /*
    out[(x * rsize[1] + y) * pi_other_dims + rp] = data[idv * pi_other_dims + rp]
    */

   // axis1, scalar:
   // out[idx0, rp] = data[idx0, idv, rp]

   

  // axis 1, 2D:
  // out[idx0, x, y, d2, ... dr-1] = data[idx0, idv, d2, .. dr-1]
  // let n_pi_other = isize[0] * isize[1] * pi_2_other.
  // let n_pi_2_other = isize[1] * pi_2_other
  // let n_pi_3_other = pi_2_other
  // out[idx0 * n_pi_other + x * n_pi_2_other + y * n_pi_3_other + rp]
  // = data[idx0 * pi_other + idv * pi_2_other + rp]
}