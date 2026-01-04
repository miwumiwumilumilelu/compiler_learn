.global main
loop:
.Lbb0:
  li a1, 0
  li a0, 0
  li a5, 1
  li a2, 0
  li a3, 10000
  slt a2, a2, a3
.Lbb1:
  slt a2, a1, a3
  bge a1, a3, .Lbb2
.Lbb3:
  li a2, 2
  remw a2, a1, a2
  li a6, 0
  xor a4, a2, a6
  seqz a4, a4
  bne a2, a6, .Lbb4
.Lbb5:
  addw a0, a0, a1
  j .Lbb6
.Lbb4:
  subw a0, a0, a1
.Lbb6:
  addw a2, a1, a5
  addw a2, a1, a5
  mv a1, a2
  j .Lbb1
.Lbb2:
  ret 


main:
.Lbb7:
  addi sp, sp, -16
  sd s0, 8(sp)
  sd ra, 0(sp)
  li a0, 16
  call _sysy_starttime
  call loop
  mv s0, a0
  li a0, 18
  call _sysy_stoptime
  mv a0, s0
  call putint
  li a0, 0
  ld s0, 8(sp)
  ld ra, 0(sp)
  addi sp, sp, 16
  ret 


