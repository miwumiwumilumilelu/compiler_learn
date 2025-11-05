%option yylineno
%{
#include "parser.tab.h"
#include "string.h"
#include "def.h"

void yyerror(const char* fmt, ...);

int yycolumn = 1;

#define YY_USER_ACTION \
    yylloc.first_line = yylloc.last_line = yylineno; \
    yylloc.first_column = yycolumn; \
    yylloc.last_column = yycolumn + yyleng - 1; \
    yycolumn += yyleng;
%}

id      [A-Za-z][A-Za-z0-9]*
int     [0-9]+
float   ([0-9]*\.[0-9]+)|([0-9]+\.)

%%
"if"      { return IF; }
"else"    { return ELSE; }
"while"   { return WHILE; }
"scan"    { return SCAN; }
"print"   { return PRINT; }
"{"       { return LC; }
"}"       { return RC; }
"("       { return LP; }
")"       { return RP; }
";"       { return SEMI; }
"="       { return ASSIGNOP; }
"+"       { return PLUS; }
"-"       { return MINUS; }
"*"       { return STAR; }
"/"       { return DIV; }
"=="      { return EQ; }
"!="      { return NE; }
">"       { return GT; }
">="      { return GE; }
"<"       { return LT; }
"<="      { return LE; }

{int}     { yylval.type_int = atoi(yytext); return INT; }
{float}   { yylval.type_float = atof(yytext); return FLOAT; }
{id}      { strncpy(yylval.type_id, yytext, 31); yylval.type_id[31] = '\0'; return ID; }

[ \t]     { /* 忽略空格和制表符 */ }
\n        { yycolumn = 1; /* 遇到换行符，重置列号 */ }
.         { yyerror("Invalid character '%s'", yytext); }

%%

int yywrap() {
    return 1;
}