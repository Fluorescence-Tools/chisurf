#ifdef _WIN32
	#include <sys\timeb.h>
#else  //probably Linux
	#include <sys/timeb.h>
#endif

class msectimer
{

public:
 msectimer ();
 float gettime();
 void reset();

private:
 timeb t;
 long secs; int msecs;

};
