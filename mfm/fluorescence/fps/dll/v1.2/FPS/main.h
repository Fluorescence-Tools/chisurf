#ifndef __MAIN_H__
#define __MAIN_H__

#include <windows.h>
#include <fpsnative.h>

/*  To use this exported function of dll, include this header
 *  in your project.
 */

#ifdef BUILD_DLL
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT __declspec(dllimport)
#endif


#ifdef __cplusplus
extern "C"
{
#endif

void DLL_EXPORT SomeFunction(const LPCSTR sometext);

void DLL_EXPORT calculate1R(double, double, double, int, double,	// linker and grid parameters
	double*, double*, double*,						// atom coordinates
	double*, int, double,							// v.d.Waals radii
	double, int,									// linker routing parameters
	unsigned char*);

#ifdef __cplusplus
}
#endif

#endif // __MAIN_H__
