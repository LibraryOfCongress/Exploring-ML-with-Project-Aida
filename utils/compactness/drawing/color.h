/*
   $Id: color.h,v 1.1.1.1 1999/10/15 12:26:21 kise Exp $
   $Date: 1999/10/15 12:26:21 $
   $Revision: 1.1.1.1 $
   $Author: kise $

	color.h
*/
#ifndef COLOR_H_INCLUDED_
#define COLOR_H_INCLUDED_

#define		BYTE           	8       /* [bit] */

#define		LINE_C 		2   /* $B@~$N?'$N#R#G#BCM$r(B 3$BHVL\$K(B
				       $B=q$/$3$H$r2>Dj(B */

#define     	BLACK       	1
#define     	WHITE       	0

/*#define		Black		0x000000
#define		White		0xffffff*/
#define		White		0x000000
#define		Black		0xffffff
#define		Red		0xff0000
#define		Green		0x00ff00
#define		Blue		0x0000ff

#define		NAMELEN        	500
#define		CLEN        	10

#define 	YES 		1
#define 	NO 		0

/* $B@~I}$N@_Dj(B
   $B$3$N>l9g$O@~$N(B1$B2hAG$r(B 5$B!_(B5 $B$K3HBg(B */
#define         WIDTH           3
#define         PWIDTH          1
#define         RWIDTH          2

#define         DEPTH           8

typedef struct{
    char *image;
    int imax, jmax;
} ImageData;

#endif /* end of COLOR_H_INCLUDED_ */
