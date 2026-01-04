.global main
square:
.Lbb0:
  fmul.s fa0, fa0, fa0
  ret 


main:
.Lbb1:
  addi sp, sp, -16
  sd ra, 8(sp)
  li a0, 1067282596
  fmv.w.x ft1, a0
  li a0, 1075671204
  fmv.w.x ft0, a0
  fadd.s ft0, ft1, ft0
  fcvt.w.s a0, ft0, rtz
  fcvt.s.w ft0, a0
  fmv.s fa0, ft0
  call square
  li a0, 1082759578
  fmv.w.x ft0, a0
  fmul.s ft0, fa0, ft0
  li a0, 1085276160
  fmv.w.x ft1, a0
  fsub.s ft0, ft0, ft1
  fcvt.w.s a0, ft0, rtz
  ld ra, 8(sp)
  addi sp, sp, 16
  ret 


