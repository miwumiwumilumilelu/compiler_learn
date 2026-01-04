.global main
add:
.Lbb0:
  mv a4, a1
  mv a5, a0
  fcvt.s.w fa2, a5
  fadd.s fa2, fa2, fa0
  fcvt.s.w fa0, a4
  fadd.s fa0, fa2, fa0
  fadd.s fa0, fa0, fa1
  ret 


main:
.Lbb1:
  addi sp, sp, -16
  sd ra, 8(sp)
  li a2, 10
  li a0, 1075838976
  fmv.w.x fa0, a0
  li a1, 20
  li a0, 1083179008
  fmv.w.x fa1, a0
  mv a0, a2
  call add
  fcvt.w.s a0, fa0, rtz
  ld ra, 8(sp)
  addi sp, sp, 16
  ret 


