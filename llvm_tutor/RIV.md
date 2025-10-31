## RIV

## RIV Pass使用——IR文件分析

**分析 Pass**，它的作用是找出在程序的每一个**基本块（Basic Block）**中，哪些整型变量是“可见”或“可达”的

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -emit-llvm -S -O1 ../inputs/input_for_riv.c -o input_for_riv.ll
```

opt操作，并调用配套的**打印 Pass** `print<riv>` 来将分析结果显示在终端上

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/opt -load-pass-plugin ./lib/libRIV.dylib --passes="print<riv>" -disable-output input_for_riv.ll
=================================================
LLVM-TUTOR: RIV analysis results
=================================================
BB id      Reachable Integer Values      
-------------------------------------------------
BB %3                                         
             i32 %0                        
             i32 %1                        
             i32 %2                        
BB %6                                         
               %4 = add nsw i32 %0, 123    
               %5 = icmp sgt i32 %0, 0     
             i32 %0                        
             i32 %1                        
             i32 %2                        
BB %17                                        
               %4 = add nsw i32 %0, 123    
               %5 = icmp sgt i32 %0, 0     
             i32 %0                        
             i32 %1                        
             i32 %2                        
BB %10                                        
               %7 = mul nsw i32 %1, %0     
               %8 = sdiv i32 %1, %2        
               %9 = icmp eq i32 %7, %8     
               %4 = add nsw i32 %0, 123    
               %5 = icmp sgt i32 %0, 0     
             i32 %0                        
             i32 %1                        
             i32 %2                        
BB %14                                        
               %7 = mul nsw i32 %1, %0     
               %8 = sdiv i32 %1, %2        
               %9 = icmp eq i32 %7, %8     
               %4 = add nsw i32 %0, 123    
               %5 = icmp sgt i32 %0, 0     
             i32 %0                        
             i32 %1                        
             i32 %2               
```

随着程序控制流的推进，可达变量的集合是不断累积和扩大的



## RIV 源码

### .h

### .c