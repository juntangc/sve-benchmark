#ifndef ARM_LIBAMATH_H
#define ARM_LIBAMATH_H

// SVE
svfloat32_t _ZGVsNxv_expf(svfloat32_t a);
svfloat32_t _ZGVsMxv_expf(svfloat32_t a, svbool_t pg);
svfloat64_t _ZGVsNxv_exp(svfloat64_t a);
svfloat64_t _ZGVsMxv_exp(svfloat64_t a, svbool_t pg);
svfloat32_t _ZGVsNxv_exp2f(svfloat32_t a);
svfloat32_t _ZGVsMxv_exp2f(svfloat32_t a, svbool_t pg);
svfloat64_t _ZGVsNxv_exp2(svfloat64_t a);
svfloat64_t _ZGVsMxv_exp2(svfloat64_t a, svbool_t pg);
svfloat32_t _ZGVsNxvv_powf(svfloat32_t a, svfloat32_t b);
svfloat32_t _ZGVsMxvv_powf(svfloat32_t a, svfloat32_t b, svbool_t pg);
svfloat64_t _ZGVsNxvv_pow(svfloat64_t a, svfloat64_t b);
svfloat64_t _ZGVsMxvv_pow(svfloat64_t a, svfloat64_t b, svbool_t pg);
svfloat32_t _ZGVsNxv_log2f(svfloat32_t a);
svfloat32_t _ZGVsMxv_log2f(svfloat32_t a, svbool_t pg);
svfloat64_t _ZGVsNxv_log2(svfloat64_t a);
svfloat64_t _ZGVsMxv_log2(svfloat64_t a, svbool_t pg);
svfloat32_t _ZGVsNxv_logf(svfloat32_t a);
svfloat32_t _ZGVsMxv_logf(svfloat32_t a, svbool_t pg);
svfloat64_t _ZGVsNxv_log(svfloat64_t a);
svfloat64_t _ZGVsMxv_log(svfloat64_t a, svbool_t pg);
svfloat32_t _ZGVsNxvv_fmodf(svfloat32_t a, svfloat32_t b);
svfloat32_t _ZGVsMxvv_fmodf(svfloat32_t a, svfloat32_t b, svbool_t pg);
svfloat64_t _ZGVsNxvv_fmod(svfloat64_t a, svfloat64_t b);
svfloat64_t _ZGVsMxvv_fmod(svfloat64_t a, svfloat64_t b, svbool_t pg);
svfloat32_t _ZGVsNxv_sinf(svfloat32_t a);
svfloat32_t _ZGVsMxv_sinf(svfloat32_t a, svbool_t pg);

// NEON
float32x4_t _ZGVnN4v_expf(float32x4_t a);
float64x2_t _ZGVnN2v_exp(float64x2_t a);
float32x4_t _ZGVnN4v_exp2f(float32x4_t a);
float64x2_t _ZGVnN2v_exp2(float64x2_t a);
float32x4_t _ZGVnN4vv_powf(float32x4_t a, float32x4_t b);
float64x2_t _ZGVnN2vv_pow(float64x2_t a, float64x2_t b);
float32x4_t _ZGVnN4v_logf(float32x4_t a);
float64x2_t _ZGVnN2v_log(float64x2_t a);
float32x4_t _ZGVnN4v_log2f(float32x4_t a);
float64x2_t _ZGVnN2v_log2(float64x2_t a);
float32x4_t _ZGVnN4vv_fmodf(float32x4_t a, float32x4_t b);
float64x2_t _ZGVnN2vv_fmod(float64x2_t a, float64x2_t b);
float32x4_t _ZGVnN4v_sinf(float32x4_t a);

// Scalar
float expf_finite_optr_aor_gcc(float a);
double exp_finite_optr_aor_gcc(double a);
float optr_aor_gcc_exp2_f32(float a);
double optr_aor_gcc_exp2_f64(double a);
float optr_aor_gcc_pow_f32(float a, float b);
double optr_aor_gcc_pow_f64(double a, double b);
double optr_aor_ac_log_f32(double a);
double optr_aor_ac_log_f64(double a);
float optr_aor_ac_log2_f32(float a);
float optr_aor_ac_log2_f64(float a);

#endif
