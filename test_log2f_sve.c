#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <arm_sve.h>
#include <arm_neon.h>
#include <armpl.h>
#include "log2f_input.h"
#include "arm_libamath.h"

#define HP_TIMING_NOW(var) \
  __asm__ __volatile__ ("isb; mrs %0, cntvct_el0" : "=r" (var))

/* Compute elapsed time in nanoseconds.  */
#undef HP_TIMING_DIFF
#define HP_TIMING_DIFF(Diff, Start, End)			\
({  uint64_t freq;						\
    __asm__ __volatile__ ("mrs %0, cntfrq_el0" : "=r" (freq));	\
   (Diff) = (uint64_t)((End) - (Start)) * ((1000000000.) / freq);	\
})

int main()
{
    int width  = 2884;
    int height = 1;
    int batch  = 100;

    float* src_data = (float *)malloc((width * height * batch) * sizeof(float));
    float* dst_data0 = (float *)malloc((width * height * batch) * sizeof(float));
    float* dst_data1 = (float *)malloc((width * height * batch) * sizeof(float));
    float* dst_data2 = (float *)malloc((width * height * batch) * sizeof(float));

    // Fill src_data 
    for(int b = 0; b < batch; b++)
    {
        for(int w = 0; w < width; w++)
        {
            src_data[b * width + w] = in0[w].arg0;
        }
    }

    float *src, *dst;
    uint64_t start, stop, diff;

    // scalar
    src = src_data;
    dst = dst_data0;
    HP_TIMING_NOW(start);
    for(int b = 0; b < batch; b++)
    {
        for(int w = 0; w < width; w++)
        {
            *dst++ = log2f(*src++);
        }
    }
    HP_TIMING_NOW(stop);
    HP_TIMING_DIFF (diff, start, stop);
    printf("log2f %g\n", (double)diff/batch/width);    

    // advanced SIMD (NEON)
    float32x4_t A, B;
    src = src_data;
    dst = dst_data1;
    HP_TIMING_NOW(start);
    for(int b = 0; b < batch; b++)
    {
        for(int w = 0; w < width; w+=4)
        {
            A = vld1q_f32(src); 
            src+=4;
            B = _ZGVnN4v_log2f(A);
            vst1q_f32(dst, B);
            dst+=4;
        }
    }
    HP_TIMING_NOW(stop);
    HP_TIMING_DIFF (diff, start, stop);
    printf("SIMD log2f %g\n", (double)diff/batch/width);    

    // SVE
    svbool_t pred;
    svfloat32_t sva, svc;
    int i = 0;
    src = src_data;
    dst = dst_data2;
    HP_TIMING_NOW(start);
    for(int b = 0; b < batch; b++, src += width, dst += width)
    {
        for(int w = 0; w < width; w+=svcntw())
        {
            pred = svwhilelt_b32(w, width);
            sva = svld1(pred, src+w);
            svc = _ZGVsMxv_log2f(sva, pred);
            svst1(pred, dst+w, svc);
        }
    }
    HP_TIMING_NOW(stop);
    HP_TIMING_DIFF (diff, start, stop);
    printf("SVE log2f %g\n", (double)diff/batch/width);    

    for(int b =0; b < batch; b++)
    {
        for(int w = 0; w < width; w++)
        {
            int idx = b * width + w;
            if((fabsf(log2f(src_data[idx]) - dst_data2[idx]) > 1.e-5f) && (b == 0))
                printf("%d %d %f %f %f %f %f\n", b, w, src_data[idx], 
                log2f(src_data[idx]), optr_aor_ac_log2_f32(src_data[idx]), 
                dst_data1[idx], dst_data2[idx]);
        }
    }

    return 0;
}
