.global main
main:
.Lbb0:
  addi sp, sp, -96
  li a0, 0
  add a0, sp, a0
  la a1, a
  li a3, 1
  li a2, 0
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 2
  li a2, 4
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 3
  li a2, 8
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 4
  li a2, 12
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 16
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 20
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 5
  li a2, 24
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 28
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 32
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 36
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 40
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 44
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 48
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 52
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 56
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 60
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 64
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 68
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 72
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 76
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 80
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 84
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 88
  add a2, a0, a2
  sw a3, 0(a2)
  li a3, 0
  li a2, 92
  add a2, a0, a2
  sw a3, 0(a2)
  li a2, 1
  li a3, 8
  mulw a2, a2, a3
  add a2, a1, a2
  li a3, 0
  li a4, 4
  mulw a3, a3, a4
  add a2, a2, a3
  lw a2, 0(a2)
  li a5, 4
  li a4, 1
  li a3, 8
  mulw a3, a4, a3
  add a3, a1, a3
  li a6, 0
  li a4, 4
  mulw a4, a6, a4
  add a3, a3, a4
  sw a5, 0(a3)
  li a5, 5
  li a4, 0
  li a3, 24
  mulw a3, a4, a3
  add a3, a0, a3
  li a4, 0
  li a6, 12
  mulw a4, a4, a6
  add a3, a3, a4
  li a6, 0
  li a4, 4
  mulw a4, a6, a4
  add a3, a3, a4
  sw a5, 0(a3)
  li a4, 1
  li a3, 8
  mulw a3, a4, a3
  add a1, a1, a3
  li a4, 0
  li a3, 4
  mulw a3, a4, a3
  add a1, a1, a3
  lw a1, 0(a1)
  addw a1, a2, a1
  li a2, 0
  li a3, 24
  mulw a2, a2, a3
  add a2, a0, a2
  li a3, 0
  li a4, 12
  mulw a3, a3, a4
  add a2, a2, a3
  li a4, 0
  li a3, 4
  mulw a3, a4, a3
  add a2, a2, a3
  lw a2, 0(a2)
  addw a1, a1, a2
  li a3, 0
  li a2, 24
  mulw a2, a3, a2
  add a0, a0, a2
  li a2, 0
  li a3, 12
  mulw a2, a2, a3
  add a0, a0, a2
  li a3, 1
  li a2, 4
  mulw a2, a3, a2
  add a0, a0, a2
  lw a0, 0(a0)
  addw a0, a1, a0
  addi sp, sp, 96
  ret 



.data
a:
  .word 1, 2, 3, 0, 4, 5, 6, 0

.bss
  .align 4
largezero:
  .space 40000
