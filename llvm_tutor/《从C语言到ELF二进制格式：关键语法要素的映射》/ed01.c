// ed01.c 
extern int val0;
extern int val1;
extern int val2;
extern void func5 ();
extern void func6 ();
extern void func9 (char *);

void func2 () { 
  val0 = 0; 
  val1 = 2; 
  val2 = 7;
}

void func3 () { 
  func5 (); 
  func6 (); 
}

void func7 () { 
  func6 ();
}

void func8 () { 
  func9 ("hello\n");
}