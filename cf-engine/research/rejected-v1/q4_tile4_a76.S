// Handwritten CF kernel, q4-tile4-v1 x exact GGML Q8_0 codes.
// GNU assembler, little-endian Linux AArch64. Only baseline ASIMD/FP and dotprod are used.
// No FP16 arithmetic: FCVTL converts binary16 storage to binary32.
// Arguments: x0=weight tile, x1=Q8 blocks, x2=block count, x3=out[4].
// Scratch: x4, v0-v7, v16-v27. v8-v15 and x19-x29 are untouched.
// Kernel source SHA identifies these candidates; this is not generated code.

    .arch armv8-a+dotprod
    .text

    .macro CF_SETUP
    movi    v0.4s, #0
    movi    v6.16b, #15
    movi    v7.16b, #8
    .endm

    .macro CF_BLOCK accumulator
    // Issue activation and scale loads before unpacking weights, keeping the
    // dependencies independent while these early loads become available.
    ldur    q4, [x1, #2]
    ldur    q5, [x1, #18]
    ld1r    {v2.4h}, [x1]
    ldr     d3, [x0]
    add     x4, x0, #8
    ldp     q16, q17, [x4]
    ldp     q18, q19, [x4, #32]
    fcvtl   v2.4s, v2.4h
    fcvtl   v3.4s, v3.4h

    and     v20.16b, v16.16b, v6.16b
    and     v21.16b, v17.16b, v6.16b
    and     v22.16b, v18.16b, v6.16b
    and     v23.16b, v19.16b, v6.16b
    ushr    v16.16b, v16.16b, #4
    ushr    v17.16b, v17.16b, #4
    ushr    v18.16b, v18.16b, #4
    ushr    v19.16b, v19.16b, #4
    sub     v20.16b, v20.16b, v7.16b
    sub     v21.16b, v21.16b, v7.16b
    sub     v22.16b, v22.16b, v7.16b
    sub     v23.16b, v23.16b, v7.16b
    sub     v16.16b, v16.16b, v7.16b
    sub     v17.16b, v17.16b, v7.16b
    sub     v18.16b, v18.16b, v7.16b
    sub     v19.16b, v19.16b, v7.16b
    movi    v24.4s, #0
    movi    v25.4s, #0
    movi    v26.4s, #0
    movi    v27.4s, #0
    sdot    v24.4s, v20.16b, v4.16b
    sdot    v25.4s, v21.16b, v4.16b
    sdot    v26.4s, v22.16b, v4.16b
    sdot    v27.4s, v23.16b, v4.16b
    fmul    v3.4s, v3.4s, v2.4s
    sdot    v24.4s, v16.16b, v5.16b
    sdot    v25.4s, v17.16b, v5.16b
    sdot    v26.4s, v18.16b, v5.16b
    sdot    v27.4s, v19.16b, v5.16b
    addp    v24.4s, v24.4s, v25.4s
    addp    v26.4s, v26.4s, v27.4s
    addp    v24.4s, v24.4s, v26.4s
    scvtf   v24.4s, v24.4s
    add     x0, x0, #72
    add     x1, x1, #34
    fmla    \accumulator\().4s, v24.4s, v3.4s
    .endm

    .p2align 4
    .global cf_q4_tile4_a76
    .type cf_q4_tile4_a76, %function
cf_q4_tile4_a76:
    CF_SETUP
    cbz     x2, .Lbaseline_done
.Lbaseline_loop:
    CF_BLOCK v0
    subs    x2, x2, #1
    b.ne    .Lbaseline_loop
.Lbaseline_done:
    str     q0, [x3]
    ret
    .size cf_q4_tile4_a76, .-cf_q4_tile4_a76

    .p2align 4
    .global cf_q4_tile4_a76_unroll2
    .type cf_q4_tile4_a76_unroll2, %function
cf_q4_tile4_a76_unroll2:
    CF_SETUP
    movi    v1.4s, #0
    cmp     x2, #2
    b.lo    .Lunroll_tail
.Lunroll_loop:
    CF_BLOCK v0
    CF_BLOCK v1
    sub     x2, x2, #2
    cmp     x2, #2
    b.hs    .Lunroll_loop
.Lunroll_tail:
    cbz     x2, .Lunroll_done
    CF_BLOCK v0
.Lunroll_done:
    // Two independent accumulation chains intentionally reassociate the sum.
    // Numerical tolerance must pass; bit identity is not an execution contract.
    fadd    v0.4s, v0.4s, v1.4s
    str     q0, [x3]
    ret
    .size cf_q4_tile4_a76_unroll2, .-cf_q4_tile4_a76_unroll2

    .p2align 4
    .global cf_q4_tile4_a76_prefetch512
    .type cf_q4_tile4_a76_prefetch512, %function
cf_q4_tile4_a76_prefetch512:
    CF_SETUP
    cbz     x2, .Lprefetch_done
.Lprefetch_loop:
    // With >=8 blocks remaining, byte +512 is inside the remaining 576+
    // weight bytes. No hint names an address beyond the caller's tile range.
    cmp     x2, #8
    b.lo    .Lprefetch_skip
    prfm    pldl1keep, [x0, #512]
.Lprefetch_skip:
    CF_BLOCK v0
    subs    x2, x2, #1
    b.ne    .Lprefetch_loop
.Lprefetch_done:
    str     q0, [x3]
    ret
    .size cf_q4_tile4_a76_prefetch512, .-cf_q4_tile4_a76_prefetch512

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
