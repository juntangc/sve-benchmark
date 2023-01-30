#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <arm_sve.h>
#include <arm_neon.h>
#include <armpl.h>
#include "pow_input.h"
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
    int width  = 1502;
    int height = 1;
    int batch  = 100;

    double* src_data = (double *)malloc((width * height * batch) * sizeof(double));
    double* src_data2 = (double *)malloc((width * height * batch) * sizeof(double));
    double* dst_data0 = (double *)malloc((width * height * batch) * sizeof(double));
    double* dst_data1 = (double *)malloc((width * height * batch) * sizeof(double));
    double* dst_data2 = (double *)malloc((width * height * batch) * sizeof(double));

    // Fill src_data 
    for(int b = 0; b < batch; b++)
    {
        for(int w = 0; w < width; w++)
        {
            src_data[b * width + w] = in0[w].arg0;
            src_data2[b * width + w] = in0[w].arg1;
        }
    }

    double *src, *src2, *dst;
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
            *dst++ = pow(*src++, *src2++);
        }
    }
    HP_TIMING_NOW(stop);
    HP_TIMING_DIFF (diff, start, stop);
    printf("pow %g\n", (double)diff/batch/width);    

    // advanced SIMD (NEON)
    float64x2_t A, A2, B;
    src = src_data;
    src2 = src_data2;
    dst = dst_data1;
    HP_TIMING_NOW(start);
    for(int b = 0; b < batch; b++)
    {
        for(int w = 0; w < width; w+=2)
        {
            A = vld1q_f64(src); 
            src+=2;
            A2 = vld1q_f64(src2); 
            src2+=2;
            B = _ZGVnN2vv_pow(A, A2);
            vst1q_f64(dst, B);
            dst+=2;
        }
    }
    HP_TIMING_NOW(stop);
    HP_TIMING_DIFF (diff, start, stop);
    printf("SIMD pow %g\n", (double)diff/batch/width);    

    // SVE
    svbool_t pred;
    svfloat64_t sva, svb, svc;
    int i = 0;
    src = src_data;
    src2 = src_data2;
    dst = dst_data2;
    HP_TIMING_NOW(start);
    for(int b = 0; b < batch; b++, src += width, dst += width)
    {
        for(int w = 0; w < width; w+=svcntd())
        {
            pred = svwhilelt_b64(w, width);
            sva = svld1(pred, src+w);
            svb = svld1(pred, src2+w);
            svc = _ZGVsMxvv_pow(sva, svb, pred);
            svst1(pred, dst+w, svc);
        }
    }
    HP_TIMING_NOW(stop);
    HP_TIMING_DIFF (diff, start, stop);
    printf("SVE pow %g\n", (double)diff/batch/width);    

    for(int b =0; b < batch; b++)
    {
        for(int w = 0; w < width; w++)
        {
            int idx = b * width + w;
            if((fabsf(pow(src_data[idx], src_data2[idx]) - dst_data2[idx]) > 1.e-5f) && (b == 0))
                printf("%d %d %f %f %f %f %f %f\n", b, w, src_data[idx], src_data2[idx], 
                pow(src_data[idx], src_data2[idx]), optr_aor_gcc_pow_f64(src_data[idx], src_data2[idx]), 
                dst_data1[idx], dst_data2[idx]);
        }
    }

    return 0;
}
