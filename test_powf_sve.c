#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <arm_sve.h>
#include <arm_neon.h>
#include <armpl.h>
#include "powf_input.h"
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
    int width  = 2501;
    int height = 1;
    int batch  = 100;

    float* src_data = (float *)malloc((width * height * batch) * sizeof(float));
    float* src_data2 = (float *)malloc((width * height * batch) * sizeof(float));
    float* dst_data0 = (float *)malloc((width * height * batch) * sizeof(float));
    float* dst_data1 = (float *)malloc((width * height * batch) * sizeof(float));
    float* dst_data2 = (float *)malloc((width * height * batch) * sizeof(float));

    // Fill src_data 
    for(int b = 0; b < batch; b++)
    {
        for(int w = 0; w < width; w++)
        {
            src_data[b * width + w] = in0[w].arg0;
            src_data2[b * width + w] = in0[w].arg1;
        }
    }

    float *src, *src2, *dst;
    uint64_t start, stop, diff;

    // scalar
    src = src_data;
    src2 = src_data2;
    dst = dst_data0;
    HP_TIMING_NOW(start);
    for(int b = 0; b < batch; b++)
    {
        for(int w = 0; w < width; w++)
        {
            *dst++ = powf(*src++, *src2++);
        }
    }
    HP_TIMING_NOW(stop);
    HP_TIMING_DIFF (diff, start, stop);
    printf("powf %g\n", (double)diff/batch/width);    

    // advanced SIMD (NEON)
    float32x4_t A, A2, B;
    src = src_data;
    src2 = src_data2;
    dst = dst_data1;
    HP_TIMING_NOW(start);
    for(int b = 0; b < batch; b++)
    {
        for(int w = 0; w < width; w+=4)
        {
            A = vld1q_f32(src); 
            src+=4;
            A2 = vld1q_f32(src2); 
            src2+=4;
            B = _ZGVnN4vv_powf(A, A2);
            vst1q_f32(dst, B);
            dst+=4;
        }
    }
    HP_TIMING_NOW(stop);
    HP_TIMING_DIFF (diff, start, stop);
    printf("SIMD powf %g\n", (double)diff/batch/width);    

    // SVE
    svbool_t pred;
    svfloat32_t sva, svb, svc;
    int i = 0;
    src = src_data;
    src2 = src_data2;
    dst = dst_data2;
    HP_TIMING_NOW(start);
    for(int b = 0; b < batch; b++, src += width, src2 += width, dst += width)
    {
        for(int w = 0; w < width; w+=svcntw())
        {
            pred = svwhilelt_b32(w, width);
            sva = svld1(pred, src+w);
            svb = svld1(pred, src2+w);
            svc = _ZGVsMxvv_powf(sva, svb, pred);
            svst1(pred, dst+w, svc);
        }
    }
    HP_TIMING_NOW(stop);
    HP_TIMING_DIFF (diff, start, stop);
    printf("SVE powf %g\n", (double)diff/batch/width);    

    for(int b =0; b < batch; b++)
    {
        for(int w = 0; w < width; w++)
        {
            int idx = b * width + w;
            if((fabsf(powf(src_data[idx], src_data2[idx]) - dst_data2[idx]) > 1.e-5f) && (b == 0))
                printf("%d %d %f %f %f %f %f %f\n", b, w, src_data[idx], src_data2[idx],
                powf(src_data[idx], src_data2[idx]), optr_aor_gcc_pow_f32(src_data[idx], src_data2[idx]), 
                dst_data1[idx], dst_data2[idx]);
        }
    }

    return 0;
}
