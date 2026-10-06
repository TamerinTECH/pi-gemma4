// Corrected reference-order kernel: q4-tile4-v1 x GGML Q8_0 codes.
// Little-endian Linux AArch64; baseline ASIMD/FP plus dotprod only.
// Match saved pinned A76 vec_dot: FOUR float lanes per output row, separate
// even/odd K-block chains, FMUL then FADD, reduce each parity only at the end.
// The odd tail intentionally matches upstream's FMUL followed by FMADD.
// Arguments x0=weights, x1=Q8, x2=blocks, x3=out. Scratch x4-x6, v0-v7,
// v16-v31; no stack or callee-preserved registers in arithmetic entrypoints.
// Rejected early-reduction/FMLA candidates are in research/rejected-v1/.

    .arch armv8-a+dotprod
    .text

    .macro CF_SETUP
    movi    v0.4s, #0
    movi    v1.4s, #0
    movi    v2.4s, #0
    movi    v3.4s, #0
    movi    v4.4s, #0
    movi    v5.4s, #0
    movi    v6.4s, #0
    movi    v7.4s, #0
    movi    v30.16b, #15
    movi    v31.16b, #8
    .endm

    .macro CF_DOTS
    ldur    q28, [x1, #2]
    ldur    q29, [x1, #18]
    ldr     x5, [x0]
    ldrh    w6, [x1]
    add     x4, x0, #8
    ldp     q16, q17, [x4]
    ldp     q18, q19, [x4, #32]
    and     v20.16b, v16.16b, v30.16b
    and     v21.16b, v17.16b, v30.16b
    and     v22.16b, v18.16b, v30.16b
    and     v23.16b, v19.16b, v30.16b
    ushr    v16.16b, v16.16b, #4
    ushr    v17.16b, v17.16b, #4
    ushr    v18.16b, v18.16b, #4
    ushr    v19.16b, v19.16b, #4
    sub     v20.16b, v20.16b, v31.16b
    sub     v21.16b, v21.16b, v31.16b
    sub     v22.16b, v22.16b, v31.16b
    sub     v23.16b, v23.16b, v31.16b
    sub     v16.16b, v16.16b, v31.16b
    sub     v17.16b, v17.16b, v31.16b
    sub     v18.16b, v18.16b, v31.16b
    sub     v19.16b, v19.16b, v31.16b
    movi    v24.4s, #0
    movi    v25.4s, #0
    movi    v26.4s, #0
    movi    v27.4s, #0
    sdot    v24.4s, v20.16b, v28.16b
    sdot    v25.4s, v21.16b, v28.16b
    sdot    v26.4s, v22.16b, v28.16b
    sdot    v27.4s, v23.16b, v28.16b
    sdot    v24.4s, v16.16b, v29.16b
    sdot    v25.4s, v17.16b, v29.16b
    sdot    v26.4s, v18.16b, v29.16b
    sdot    v27.4s, v19.16b, v29.16b
    fmov    d16, x5
    dup     v17.4h, w6
    fcvtl   v16.4s, v16.4h
    fcvtl   v17.4s, v17.4h
    add     x0, x0, #72
    add     x1, x1, #34
    .endm

    .macro CF_ACCUM a, b, c, d
    CF_DOTS
    fmul    v18.4s, v16.4s, v17.4s
    scvtf   v24.4s, v24.4s
    scvtf   v25.4s, v25.4s
    scvtf   v26.4s, v26.4s
    scvtf   v27.4s, v27.4s
    fmul    v24.4s, v24.4s, v18.s[0]
    fmul    v25.4s, v25.4s, v18.s[1]
    fmul    v26.4s, v26.4s, v18.s[2]
    fmul    v27.4s, v27.4s, v18.s[3]
    fadd    v\a\().4s, v\a\().4s, v24.4s
    fadd    v\b\().4s, v\b\().4s, v25.4s
    fadd    v\c\().4s, v\c\().4s, v26.4s
    fadd    v\d\().4s, v\d\().4s, v27.4s
    .endm

    .macro CF_FUNCTION name, prefetch
    .p2align 4
    .global \name
    .type \name, %function
\name:
    CF_SETUP
    cmp     x2, #2
    b.lo    2f
1:
    .if \prefetch
    cmp     x2, #8
    b.lo    4f
    prfm    pldl1keep, [x0, #512]
4:
    .endif
    CF_ACCUM 0, 1, 2, 3
    CF_ACCUM 4, 5, 6, 7
    sub     x2, x2, #2
    cmp     x2, #2
    b.hs    1b
2:
    // Each row reduces (lane0+lane1)+(lane2+lane3), separately per parity.
    faddp   v16.4s, v0.4s, v1.4s
    faddp   v17.4s, v2.4s, v3.4s
    faddp   v16.4s, v16.4s, v17.4s
    faddp   v17.4s, v4.4s, v5.4s
    faddp   v18.4s, v6.4s, v7.4s
    faddp   v17.4s, v17.4s, v18.4s
    fadd    v0.4s, v17.4s, v16.4s
    cbz     x2, 3f
    // The pinned upstream odd tail reduces integer lanes before converting,
    // multiplies by weight scale first, then fuses activation scale + sumf.
    CF_DOTS
    addp    v24.4s, v24.4s, v25.4s
    addp    v26.4s, v26.4s, v27.4s
    addp    v24.4s, v24.4s, v26.4s
    scvtf   v24.4s, v24.4s
    fmul    v24.4s, v24.4s, v16.4s
    fmla    v0.4s, v24.4s, v17.4s
3:
    str     q0, [x3]
    ret
    .size \name, .-\name
    .endm

    CF_FUNCTION cf_q4_tile4_a76, 0
    // The reference algorithm itself requires two independent parity chains.
    // Legacy unroll2 now aliases the corrected baseline, not a distinct win.
    .global cf_q4_tile4_a76_unroll2
    .type cf_q4_tile4_a76_unroll2, %function
    .set cf_q4_tile4_a76_unroll2, cf_q4_tile4_a76
    // One bounded +512 hint per pair when >=8 blocks remain. Arithmetic is
    // identical to baseline; the old per-block hint schedule is archived.
    CF_FUNCTION cf_q4_tile4_a76_prefetch512, 1

    .macro CF_CHECK_GPR reg, bit
    cmp     x\reg, #\reg
    cset    x9, ne
    orr     x0, x0, x9, lsl #\bit
    .endm
    .macro CF_CHECK_D reg, bit
    fmov    x9, d\reg
    cmp     x9, #\reg
    cset    x9, ne
    orr     x0, x0, x9, lsl #\bit
    .endm

    .p2align 4
    .global cf_q4_tile4_a76_abi_probe
    .type cf_q4_tile4_a76_abi_probe, %function
cf_q4_tile4_a76_abi_probe:
    // Save the caller's original nonvolatile registers, then put distinct
    // sentinels into them before calling the requested arithmetic candidate.
    stp     x29, x30, [sp, #-160]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8, d9, [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #128]
    stp     d14, d15, [sp, #144]
    mov     x19, #19
    mov     x20, #20
    mov     x21, #21
    mov     x22, #22
    mov     x23, #23
    mov     x24, #24
    mov     x25, #25
    mov     x26, #26
    mov     x27, #27
    mov     x28, #28
    mov     x29, #29
    mov     x9, #8
    fmov    d8, x9
    mov     x9, #9
    fmov    d9, x9
    mov     x9, #10
    fmov    d10, x9
    mov     x9, #11
    fmov    d11, x9
    mov     x9, #12
    fmov    d12, x9
    mov     x9, #13
    fmov    d13, x9
    mov     x9, #14
    fmov    d14, x9
    mov     x9, #15
    fmov    d15, x9
    blr     x4
    mov     x0, #0
    CF_CHECK_GPR 19, 0
    CF_CHECK_GPR 20, 1
    CF_CHECK_GPR 21, 2
    CF_CHECK_GPR 22, 3
    CF_CHECK_GPR 23, 4
    CF_CHECK_GPR 24, 5
    CF_CHECK_GPR 25, 6
    CF_CHECK_GPR 26, 7
    CF_CHECK_GPR 27, 8
    CF_CHECK_GPR 28, 9
    CF_CHECK_GPR 29, 10
    CF_CHECK_D 8, 11
    CF_CHECK_D 9, 12
    CF_CHECK_D 10, 13
    CF_CHECK_D 11, 14
    CF_CHECK_D 12, 15
    CF_CHECK_D 13, 16
    CF_CHECK_D 14, 17
    CF_CHECK_D 15, 18
    ldp     d8, d9, [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     x29, x30, [sp], #160
    ret
    .size cf_q4_tile4_a76_abi_probe, .-cf_q4_tile4_a76_abi_probe

    .section .note.GNU-stack,"",%progbits
