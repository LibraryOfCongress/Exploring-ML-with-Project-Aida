/*
  $Id: defs.h,v 1.1.1.1 1999/10/15 12:40:27 kise Exp $
  $Date: 1999/10/15 12:40:27 $
  $Revision: 1.1.1.1 $
  $Author: kise $
*/

#ifndef DEFS_H_INCLUDED_
#define DEFS_H_INCLUDED_
#include "const.h"

namespace voronoi{
    typedef unsigned short Coordinate;
    // Faisal Shafait's modification: changed Label type from short to int
    typedef unsigned int   Label;
    typedef unsigned int   NumPixel;
    typedef unsigned int   Key;
    typedef unsigned int   HashVal;
    typedef double         Coord;

    // Structure to represent a min heap node 
    struct MinHeapNode 
    { 
        int  v; 
        int dist; 
    }; 
      
    // Structure to represent a min heap 
    struct MinHeap 
    { 
        int size;      // Number of heap nodes present currently 
        int capacity;  // Capacity of min heap 
        int *pos;     // This is needed for decreaseKey() 
        struct MinHeapNode **array; 
    }; 

    /* Representation of graphs - start */
    // A structure to represent an adjacency list node 
    struct AdjListNode 
    { 
        int dest; 
        int weight;
        int lineseg_idx;
        struct AdjListNode* next; 
    }; 
      
    // A structure to represent an adjacency list 
    struct AdjList 
    { 
        struct AdjListNode *head;  
    }; 
      
    // A structure to represent a graph. A graph 
    // is an array of adjacency lists. 
    // Size of array will be V (number of vertices  
    // in graph) 
    struct Graph 
    { 
        int V; 
        struct AdjList* array; 
    }; 
      

    /* The structure for a Voronoi point */
    struct CC{
        int lab;
        struct CC *next;
    };
    /* The structure for zone */
    typedef struct {
        int len;
        int numOfCCs;
        struct CC* cc_head;
        struct AdjListNode* head; 
    } Zone;
    
    /* Representation of graphs - end */
    
    /* The structure for a vector */
    typedef struct{
        int x;
        int y;
    } Vector;

    /* The structure for a binary image */ 
    typedef struct{
        char *image;
        Coordinate imax, jmax; /* width and height of the image */
    } ImageData;

    typedef struct{
        char *filename;
        int width,height;
        int numOfCC;
        int numOfSites;
        float IQR, Q1, Q3, sizeMean, sizeStd;
        float angleMode;
        int td1;
        float td2;
    } MetaData;

    /* The structure for a pixel */
    typedef struct{
        Label label;     /* label */
        Coordinate xax,yax; /* coordinates */
    } BlackPixel;

    /* The structure for a shape of CC */
    typedef struct{
         Coordinate x_min,x_max,y_min,y_max;
         float conf[2048];
         float conf_iqr;
         float conf_median;
         int conf_idx;
    } Shape;


    /* A structure representing the barycentric coordinates of each connected component and the vertical and horizontal lengths of the rectangle surrounding it */
    /*
    typedef struct{
      unsigned short x,y;
      unsigned short dx,dy;
      unsigned int bpn;
    } Component;
    */

    /* The structure for a neighboring relation
       between connected components(CC's)

       lab2
       -------
       |  x  |
       -------
       --- /
       | |/ angle
       |x|-----
       | |
       ---
       lab1
    */
    typedef struct{
        float dist;       /* min. distance between CC's */
        float angle;      /* angle between CC's */
        Label lab1, lab2; /* labels of CC's */
    } Neighborhood;

    /* The structure for a Voronoi edge */
    typedef struct edge_node{
        int	sp,ep;
        float conf;
        int weight;
        Coordinate xs,xe,ys,ye;  /* + (xs,ys)
                                    \
                                    \  Voronoi edge
                                    \
                                    + (xe,ye)
                                 */
        Label lab1,lab2;  /* this Voronoi edge is between
                             CC of a label "lab1" and that of lab2 */
        unsigned short yn;

        struct edge_node *next;
        int lineseg_idx;
    } LineSegment;

    /* The structure for a Voronoi point */
    typedef struct node{
        int line;
        struct node *next;
    } EndPoint;

    /* The structure for a hash table for
       representing labels of CC's */
    typedef struct hash {
        Label lab1;
        Label lab2;
        unsigned int entry;
        struct hash *next;
    } HashTable;
    /*
      typedef struct hash {
      unsigned long id;
      unsigned int entry;
      struct hash *next;
      } HashTable;
    */

    /* The structure for a rectangle */
    typedef struct{
        Coordinate is,ie,js,je;
    } Rect;

    struct Freenode {
        struct Freenode	*nextfree;
    };

    struct Freelist {
        struct Freenode	*head;
        // Faisal Shafait's modification to fix memory leak
#ifdef h_iupr_
        colib::narray<void *> allocated_chunks;
#endif
        // End of Modification
        int		nodesize;
    };

    struct Point {
        float		x,y;
    };

    /* structure used both for sites and for vertices */
    struct Site {
        struct Point	coord;
        int		sitenbr;
        int		refcnt;
        unsigned int	label;
    };

    struct Edge	{
        float		a,b,c;
        struct	Site 	*ep[2];
        struct	Site	*reg[2];
        int		edgenbr;
        unsigned int	lab1,lab2;
        float		dist;
    };

    struct Halfedge {
        struct Halfedge	*ELleft, *ELright;
        struct Edge	*ELedge;
        int		ELrefcnt;
        char		ELpm;
        struct Site	*vertex;
        float		ystar;
        struct Halfedge	*PQnext;
    };
}
#endif /* DEFS_H_INCLUDED_ */
