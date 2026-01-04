.global main
main:
.Lbb0:
  la a1, count
  li a0, 7
.Lbb1:
  li a3, 1
  xor a2, a0, a3
  snez a2, a2
  beq a0, a3, .Lbb2
.Lbb3:
  lw a2, 0(a1)
  li a3, 1
  addw a2, a2, a3
  sw a2, 0(a1)
  li a2, 2
  remw a2, a0, a2
  li a4, 0
  xor a3, a2, a4
  seqz a3, a3
  bne a2, a4, .Lbb4
.Lbb5:
  li a2, 2
  divw a0, a0, a2
  j .Lbb1
.Lbb4:
  li a2, 3
  mulw a0, a0, a2
  li a2, 1
  addw a0, a0, a2
  j .Lbb1
.Lbb2:
  lw a0, 0(a1)
  ret 



.data

.bss
  .align 4
count:
  .space 4
