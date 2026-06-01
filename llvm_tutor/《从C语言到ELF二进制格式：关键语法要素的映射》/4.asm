
nonpic_reldemo:	file format elf64-sparc

Disassembly of section .text:

0000000000200158 <func2>:
  200158: 9d e3 bf 80  	save %sp, -0x80, %sp
  20015c: 31 00 00 00  	sethi 0x0, %i0
  200160: b0 06 23 00  	add %i0, 0x300, %i0
  200164: b1 2e 30 0c  	sllx %i0, 0xc, %i0
  200168: b2 10 20 09  	mov	0x9, %i1
  20016c: f2 26 21 8c  	st %i1, [%i0+0x18c]
  200170: 81 c7 e0 08  	ret
  200174: 81 e8 00 00  	restore

0000000000200178 <func3>:
  200178: 9d e3 bf 50  	save %sp, -0xb0, %sp
  20017c: 7f ff ff f7  	call 0x200158
  200180: 01 00 00 00  	nop
  200184: 81 c7 e0 08  	ret
  200188: 81 e8 00 00  	restore
