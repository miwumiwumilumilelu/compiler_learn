.global main
main:
.Lbb0:
  li a2, 0
  li a0, 0
  mv a1, a2
.Lbb1:
  li a3, 10
  slt a2, a0, a3
  bge a0, a3, .Lbb2
.Lbb3:
  li a3, 3
  xor a2, a0, a3
  seqz a2, a2
  addw a1, a1, a0
  li a3, 20
  slt a2, a3, a1
  li a2, 1
  addw a0, a0, a2
  j .Lbb1
.Lbb2:
  mv a0, a1
  ret 


