.global main
main:
.Lbb0:
  li a3, 100
  li a2, 2
  li a0, 0
  li a1, 0
  slt a1, a1, a3
.Lbb1:
  slt a1, a0, a3
  bge a0, a3, .Lbb2
.Lbb3:
  addw a1, a0, a2
  addw a0, a0, a2
  j .Lbb1
.Lbb2:
  ret 


