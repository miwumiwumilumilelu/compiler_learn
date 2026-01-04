
#include <stdio.h>
#include <sys/time.h>

void putint(int a) {
    printf("%d", a);
}

void putch(int a) {
    printf("%c", a);
}

void putfloat(float a) {
    printf("%f", a);
}

int getint() {
    int t;
    scanf("%d", &t);
    return t;
}

int getch() {
    char c;
    scanf("%c", &c);
    return (int)c;
}

int getfloat() {
    float t;
    scanf("%f", &t);
    return (int)t; // 简化的转换
}

void _sysy_starttime(int lineno) {
    // 简单实现：打印日志或什么都不做
}

void _sysy_stoptime(int lineno) {
    // 简单实现
}
