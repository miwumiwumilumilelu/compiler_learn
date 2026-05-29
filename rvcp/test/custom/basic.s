.global main
loop:
.Lbb0:
  mv a3, a2
  mv a4, a1
  mv a5, a0
  li a0, 0
  li a1, 0
  fmv.w.x fa0, a1
  li a2, 1
  li a1, 0
  slt a1, a1, a3
.Lbb1:
  slt a1, a0, a3
  bge a0, a3, .Lbb2
.Lbb3:
  li a1, 4
  mulw a1, a0, a1
  add a1, a5, a1
  flw fa1, 0(a1)
  li a1, 4
  mulw a1, a0, a1
  add a1, a4, a1
  flw fa2, 0(a1)
  fmul.s fa1, fa1, fa2
  fadd.s fa0, fa0, fa1
  addw a1, a0, a2
  addw a1, a0, a2
  mv a0, a1
  j .Lbb1
.Lbb2:
  ret 


main:
.Lbb4:
  li t0, 40112
  sub sp, sp, t0
  li t6, 40104
  add t6, t6, sp
  sd s0, 0(t6)
  li t6, 40096
  add t6, t6, sp
  sd s1, 0(t6)
  li t6, 40088
  add t6, t6, sp
  sd s2, 0(t6)
  li t6, 40080
  add t6, t6, sp
  sd s3, 0(t6)
  li t6, 40072
  add t6, t6, sp
  sd s4, 0(t6)
  li t6, 40064
  add t6, t6, sp
  sd s5, 0(t6)
  li t6, 40056
  add t6, t6, sp
  sd s6, 0(t6)
  li t6, 40048
  add t6, t6, sp
  sd s7, 0(t6)
  li t6, 40040
  add t6, t6, sp
  fsd fs0, 0(t6)
  li t6, 40032
  add t6, t6, sp
  fsd fs1, 0(t6)
  li t6, 40024
  add t6, t6, sp
  fsd fs2, 0(t6)
  li t6, 40016
  add t6, t6, sp
  fsd fs3, 0(t6)
  li t6, 40008
  add t6, t6, sp
  sd ra, 0(t6)
  li a0, 0
  add s3, sp, a0
  li a0, 20000
  add s5, sp, a0
  la s0, COUNT
  li s2, 0
  li s1, 0
  li s4, 5000
  li a0, 0
  fmv.w.x fs3, a0
  li a0, 0
  fmv.w.x fs2, a0
  li a0, 1065353216
  fmv.w.x fs0, a0
  li a0, 22
  call _sysy_starttime
  li s7, 1
  li a0, 0
  lw s6, 0(s0)
  slt a0, a0, s6
.Lbb5:
  slt a0, s2, s6
  bge s2, s6, .Lbb6
.Lbb7:
  li s0, 0
  li a0, 10
  remw a0, s2, a0
  beq a0, zero, .Lbb8
.Lbb9:
  li a0, 0
  fmv.w.x ft0, a0
  li a0, 1065353216
  fmv.w.x fs0, a0
  fmv.s fs1, ft0
  j .Lbb10
.Lbb8:
  li a0, 1036831949
  fmv.w.x ft0, a0
  fadd.s fs1, fs2, ft0
  li a0, 1045220557
  fmv.w.x ft0, a0
  fadd.s fs0, fs0, ft0
.Lbb10:
  li a1, 1
  slt a0, s1, s4
.Lbb11:
  slt a0, s0, s4
  bge s0, s4, .Lbb12
.Lbb13:
  fcvt.s.w ft0, s0
  fadd.s ft0, fs1, ft0
  li a0, 4
  mulw a0, s0, a0
  add a0, s3, a0
  fsw ft0, 0(a0)
  fcvt.s.w ft0, s0
  fadd.s ft0, fs0, ft0
  li a0, 4
  mulw a0, s0, a0
  add a0, s5, a0
  fsw ft0, 0(a0)
  addw a0, s0, a1
  addw s0, s0, a1
  j .Lbb11
.Lbb12:
  mv a0, s3
  mv a1, s5
  mv a2, s4
  call loop
  fadd.s ft0, fs3, fa0
  addw a0, s2, s7
  addw a0, s2, s7
  mv s2, a0
  mv s1, s0
  fmv.s fs3, ft0
  fmv.s fs2, fs1
  j .Lbb5
.Lbb6:
  li a0, 40
  call _sysy_stoptime
  li a0, 1463647628
  fmv.w.x ft0, a0
  fsub.s ft0, fs3, ft0
  li a0, 897988541
  fmv.w.x ft1, a0
  fle.s a0, ft0, ft1
  beq a0, zero, .Lbb14
.Lbb15:
  li a0, 897988541
  fmv.w.x ft1, a0
  