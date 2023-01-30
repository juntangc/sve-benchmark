#include <math.h>
#include "bench_one.h"
#include <arm_sve.h>
#include <armpl.h>

svfloat32_t _ZGVsNxv_expf(svfloat32_t);
svfloat32_t _ZGVsMxv_expf(svfloat32_t a, svbool_t pg);

float *abufsp, *bbufsp;
float *obufsp;

int main(int argc, char **argv) {

  uint64_t start, stop, diff;
  abufsp = (float *)malloc(NITER1*sizeof(float));
  bbufsp = (float *)malloc(NITER1*sizeof(float));
  obufsp = (float *)malloc(NITER1*sizeof(float));
  srandom(time(NULL));

  fillSP(abufsp, -100, 100);
  fillSP(bbufsp, -100, 100);
  
  HP_TIMING_NOW(start);
  uint64_t t = currentTimeMicros();
  for(int j=0;j<NITER2;j++) {
	float *p = (float *)(abufsp);
	float *q = (float *)(obufsp);
    for(int i=0;i<NITER1;i++) 
      *q++ = expf(*p++);
  }

  printf("%f ns per computation\n", (double)(currentTimeMicros() - t) / NITER * 1000.0);
  HP_TIMING_NOW(stop);
  HP_TIMING_DIFF (diff, start, stop);
  //printf("%f ns per computation, hp_timing\n", (double)diff / NITER * 1000.0);

  svfloat32_t x = svdup_f32(4.0);
  svbool_t p_all = svptrue_b32();

  svfloat32_t y = _ZGVsMxv_expf(x, p_all);

  return 0;
}

