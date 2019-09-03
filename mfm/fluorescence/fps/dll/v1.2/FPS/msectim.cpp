#include "msectim.h"

msectimer::msectimer ()
 {
  reset();
 }

void msectimer::reset()
 {
  ftime(&t);
  secs = (int)t.time; msecs = t.millitm;
 }

float msectimer::gettime()
 {
  ftime(&t);
  return (t.time-secs)+0.001f*(t.millitm-msecs);
 }
