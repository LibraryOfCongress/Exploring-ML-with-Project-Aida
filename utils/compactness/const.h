/*
  $Id: const.h,v 1.1.1.1 1999/10/15 12:40:27 kise Exp $
  $Date: 1999/10/15 12:40:27 $
  $Revision: 1.1.1.1 $
  $Author: kise $
*/
namespace voronoi{
#ifndef NULL
#define NULL  0
#endif
#define DELETED   -2

#define BYTE    8       /* [bit] */
#define BLACK   1
#define WHITE   0
#define NAMELEN   500

    /* initial size and increments */
#define INITPIXEL 5000000  /* initial size of black pixels */
#define INCPIXEL        100000   /* increments of black pixels */
#define INITNEIGHBOR  50000    /* initial size of the array "neighbor" */
#define INCNEIGHBOR     10000    /* its increments */

#define INITLINE        50000    /* initial size of the array "lineseg" */
#define INCLINE         10000    /* its increments */
#define SITE_BOX        10000

    /* MAX values */
#define UNSIGNED_MAX    0xFFFFFFFF /* max. of unsigned int */
#define IMG_IMAX        10000
    // Faisal Shafait's modification: changed NOLABEL to max_int instead of max_short
#define NOLABEL   0xFFFFFFFF /* max. of unsigned int */
#define LABELMAX        NOLABEL

#define YES   1
#define NO    0
#define RIGHTANGLE  90
#define M_PI    3.14159265358979323846

    /* for sweep2 */
#define LE    0
#define RE    1
#define DELETED   -2

    /* An end point of a Voronoi edge is on the border of an image,
       its sitenbr is FRAME (erase.c, output.c) */
#define FRAME   -1

    /* for hash table */
#define M1    10007 /* prime number */
#define M2    3 /* prime number */
#define NIL   0
#define NODATA    -1

    /* default values */
#define SAMPLE_RATE   13
#define NOISE_MAX     15
#define FREQ_RATE     0.34
#define Ta_CONST      40
#define SMWIND        2    /* for smoothing the freq. distribution of distance */
    /* VORONOI++ */
#define ANGLE_SIGMA   0.52 /* 30 degree */
#define ANGLE_K       0.5
    /* VORONOI SHAPE */
#define SIZE_SIM      5      // fixed
#define ESP           0.001  // fixed
#define CONF_LEVEL    1.0    // fixed
#define Ts_CONST      20     // 
#define SHAPE_K       0.00625  // Depends on variation of size of CCs

    /* ボロノイ辺の出力情報 */
#define OUTPUT    1
#define NO_OUTPUT 0

#define TRUE    1
#define FALSE   2

#define ZONEMAX   1000

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
}
