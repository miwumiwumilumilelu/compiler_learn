.global main
fib:
.Lbb0:
  addi sp, sp, -32
  sd s0, 24(sp)
  sd s1, 16(sp)
  sd ra, 8(sp)
  mv s0, a0
  li a1, 2
  slt a0, s0, a1
  bge s0, a1, .Lbb1
.Lbb2:
  mv a0, s0
  j .Lbb3
.Lbb1:
  li a0, 1
  subw a0, s0, a0
  call fib
  mv s1, a0
  li a0, 2
  subw a0, s0, a0
  call fib
  addw a0, s1, a0
.Lbb3:
  ld s0, 24(sp)
  ld s1, 16(sp)
  ld ra, 8(sp)
  addi sp, sp, 32
  ret 


main:
.Lbb4:
  addi sp, sp, -16
  sd ra, 8(sp)
  li a0, 5
  call fib
  ld ra, 8(sp)
  addi sp, sp, 16
  ret 


