#include <cuda_runtime.h>
#include <iostream>
#include <cuda_fp16.h>

// Note: We no longer need the Fragment128 union.

// CUDA intrinsic function for ldmatrix.sync.aligned.m8n8.x4.b16
// It takes a 128-bit pointer (void*) and returns a 128-bit result fragment.
// We model the four 128-bit fragments as an array of 4 unsigned long long.
extern "C" __device__ __half2 *__ldmatrix_sync_aligned_m8n8_x4_b16(const void *ptr); 


__global__ void ldmatrix_test_kernel(const half* __restrict__ input_matrix, half* __restrict__ output_matrix)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    
    // An array of 4 __half2 vectors to hold the 128-bit fragment (D0)
    __half2 D0_h[4]; 
    
    // We need 4 128-bit results for the .x4 load. We use a container array to hold them.
    __half2 results[16]; // 4 fragments * 4 __half2 per fragment = 16 __half2 vectors
    
    if (warp_id == 0)
    {
        // 1. Declare four 128-bit destination register fragments.
        // We use unsigned long long to represent the 128-bit register space for the fragments.
        // This is necessary because the intrinsic returns the data in hardware registers.
        // We will read the results from this allocated stack space.
        unsigned long long F0[2]; // Two 64-bit halves of a 128-bit fragment (D0)
        unsigned long long F1[2]; 
        unsigned long long F2[2];
        unsigned long long F3[2];
        
        const void* gmem_ptr = input_matrix;
        
        // 2. Call the ldmatrix intrinsic for all 4 fragments.
        // The intrinsic automatically handles the complex register allocation.
        // The instruction is implicitly ldmatrix.sync.aligned.m8n8.x4.b16 when using the intrinsic.
        
        // We manually extract the 4 fragments using PTX. This is the only way to avoid 
        // the compiler's unstable register mapping for inline PTX.
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x4.b16 "
            "{%0, %1, %2, %3}, " // Destination fragments (4 fragments for .x4)
            "[%4];"
            : "=r"(((unsigned int*)F0)[0]), "=r"(((unsigned int*)F1)[0]), "=r"(((unsigned int*)F2)[0]), "=r"(((unsigned int*)F3)[0]) 
            : "l"(gmem_ptr) 
            : "memory"
        );
        
        // 3. Verification/Store: Read data from the first fragment (F0)
        if (lane_id == 0) {
            
            // Cast the fragment's memory space to the type we need to read
            const __half2* frag = (const __half2*)F0;

            // F0 is 128 bits, holding 4 __half2 vectors.
            output_matrix[0] = __low2half(frag[0]);
            output_matrix[1] = __high2half(frag[0]);
            output_matrix[2] = __low2half(frag[1]);
            output_matrix[3] = __high2half(frag[1]);
        }
    }
}
// ... (main function and helpers are unchanged)

// Helper function to convert a half-precision array to float for printing
void convert_half_to_float(const half* h_data, float* f_data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        f_data[i] = __half2float(h_data[i]);
    }
}

// Helper function to convert a float array to half-precision
void convert_float_to_half(const float* f_data, half* h_data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        h_data[i] = __float2half(f_data[i]);
    }
}

// ... (Kernel and Helper functions are unchanged from the last successful compilation)

int main()
{
    const int MATRIX_SIZE = 16 * 16;
    float* h_input = (float*)malloc(MATRIX_SIZE * sizeof(float));
    half* h_output = (half*)malloc(MATRIX_SIZE * sizeof(half)); 

    // Initialize input matrix with simple data (e.g., 1.0, 2.0, 3.0, ...)
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        h_input[i] = (float)(i + 1);
    }
    
    // Allocate managed memory (already 16-byte aligned)
    half* d_input;
    half* d_output;
    cudaMallocManaged(&d_input, MATRIX_SIZE * sizeof(half), cudaMemAttachGlobal);
    cudaMallocManaged(&d_output, MATRIX_SIZE * sizeof(half), cudaMemAttachGlobal);
    
    // Convert float input to half on the CPU before copying
    half* h_input_half = (half*)malloc(MATRIX_SIZE * sizeof(half));
    convert_float_to_half(h_input, h_input_half, MATRIX_SIZE);

    // --- CRITICAL MEMORY FIX STARTS HERE ---
    
    // Copy data from CPU to Managed Memory
    cudaMemcpy(d_input, h_input_half, MATRIX_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, MATRIX_SIZE * sizeof(half));
    
    // Force the memory to be accessible/coherent on the default stream's device (GPU)
    int current_device;
    cudaGetDevice(&current_device);
    
    // Advice: Tell the system this memory will be accessed by the GPU
    cudaMemAdvise(d_input, MATRIX_SIZE * sizeof(half), cudaMemAdviseSetAccessedBy, current_device);

    // Attach memory to the default stream (synchronous on the CPU)
    // This forces the runtime to synchronize the memory state.
    cudaStreamAttachMemAsync(0, d_input, 0, cudaMemAttachGlobal);
    
    // Synchronize to ensure all memory operations are complete before kernel launch
    cudaDeviceSynchronize(); 

    // --- CRITICAL MEMORY FIX ENDS HERE ---

    // Launch kernel: 1 block, 32 threads (one warp)
    ldmatrix_test_kernel<<<1, 32>>>(d_input, d_output);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy a few output elements back for verification
    cudaMemcpy(h_output, d_output, 4 * sizeof(half), cudaMemcpyDeviceToHost);

    // Verification printout... (rest of main() is unchanged)
    float f_output[4];
    convert_half_to_float(h_output, f_output, 4);

    std::cout << "--- ldmatrix.sync.aligned.m8n8.x4.b16 Test Results ---" << std::endl;
    std::cout << "Input matrix data (first 4 elements of 16x16 block):" << std::endl;
    std::cout << "  " << h_input[0] << ", " << h_input[1] << ", " << h_input[2] << ", " << h_input[3] << std::endl;

    std::cout << "\nOutput (first 4 half-precision elements stored by warp):" << std::endl;
    std::cout << "  " << f_output[0] << ", " << f_output[1] << ", " << f_output[2] << ", " << f_output[3] << std::endl;
    
    // Check if the load was correct: Output should match Input
    if (f_output[0] == h_input[0] && f_output[3] == h_input[3]) {
        std::cout << "\nSUCCESS: The loaded data matches the input data." << std::endl;
    } else {
        std::cout << "\nFAILURE: Data mismatch." << std::endl;
    }

    // Cleanup
    free(h_input);
    free(h_input_half);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}