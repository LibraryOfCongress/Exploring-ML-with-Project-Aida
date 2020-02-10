/*
  $Date: 1999/10/15 12:40:27 $
  $Revision: 1.1.1.1 $
  $Author: kise $
  img_to_site.c
  
    A program that extracts sampled boundary lines and labels attached 
    to each sampling point from the input image and sets the 
    coordinates x, y and label ln of the sampling point to Site type sites

    Input file
    Image in sunraster format

    Output
    Site type sites
*/

#include <stdio.h>
#include "defs.h"
#include "const.h"
#include "function.h"
#include "extern.h"


namespace voronoi{
    /* Vecor Definition of type labels */
    Vector UpLeft = {-1,-1};
    Vector Up = {0,-1};
    Vector UpRight = {1,-1};
    Vector Left = {-1,0};
    Vector Right = {1,0};
    Vector DownLeft = {-1,1};
    Vector Down = {0,1};
    Vector DownRight = {1,1};
    Vector Center = {0,0};

#define vector_equal(v1,v2) ((v1.x==v2.x) && (v1.y==v2.y))

    /*
      Site extraction using boundary tracking type labeling (Image is not boundary tracking when edge is 0 and coordinates are edge of image.)
    */

    void img_to_site(ImageData *imgd,MetaData *metadata)
    {
        Coordinate imax = imgd->imax;
        Coordinate jmax = imgd->jmax;
        int i,j,k,b;
        Label lab[3],ln;
        char byte;
        int  index=0;

        /* Initialization */
        sample_pix=0;

        /*
          for(i=0;i<MAXPIXEL;i++){
          ((bpx+i)->label)=0;
          ((bpx+i)->xax)=0;
          ((bpx+i)->yax)=0;
          }
        */
  
        lab_format(imgd);
  
        /* Main process from here */
        ln = -1;
        BPnbr =0;
        int current_nsites=0;
        for(j = 1 ; j < jmax-1 ; j++){
            for(b = 0 ; b < imax/BYTE ; b++) {
                byte=*(imgd->image+(imax/BYTE)*j+b);
                if(byte!=0x00){
                    for(k = byte_pos(byte,0) ; k < BYTE ; k = byte_pos(byte,++k)){
                        i=BYTE*b+k;
                        *lab=lab_get(i-1,j);          /* Left Label */
                        *(lab+1)=lab_get(i-1,j-1);  /* UpLeft Label */
                        *(lab+2)=lab_get(i,j);        /* Center Label */
                        if(*(lab+2)==NOLABEL){
                            if(*lab==NOLABEL&&*(lab+1)==NOLABEL){
                                ln++;
                                if(ln>=LABELMAX) {
                                    fprintf(stderr,"table の領域が足りません\n");
                                    fprintf(stderr,"LABELMAX の値を再設定して下さい\n");
                                    exit(1);
                                }
                                current_nsites = nsites;
                                bf_edgelab_smpl(imgd,i,j,ln);
                                // Modification by Faisal Shafait
                                // keep track of noise components to remove them
                                // from the output image
                                if(ln>=nconcomp_size){
                                    noise_comp=(bool *)myrealloc(noise_comp,nconcomp_size,nconcomp_inc,sizeof(bool));
                                    nconcomp_size+=nconcomp_inc;
                                }
                                if(current_nsites == nsites){
                                    noise_comp[ln]=true;
                                }else{
                                    noise_comp[ln]=false;
                                }
                                // End of Modification
                            }
                            else if(*lab!=NOLABEL){
                                lab_set(i,j,*lab);
                                bpxset(i,j,*lab);
                            }
                            else{
                                bf_edgelab_smpl(imgd,i,j,*(lab+1));
                            }
                        }
                    }
                }
            }
        }
              
        LABELnbr = ++ln; /* Number of connected components */
        
        qsort(sites, nsites, sizeof *sites, scomp);
        xmin=sites[0].coord.x; 
        xmax=sites[0].coord.x;
        for(i=1; i<nsites; i+=1) {
            if(sites[i].coord.x < xmin) xmin = sites[i].coord.x;
            if(sites[i].coord.x > xmax) xmax = sites[i].coord.x;
        }
        ymin = sites[0].coord.y;
        ymax = sites[nsites-1].coord.y;
        for(i=0;i<nsites;i++){
            sites[i].sitenbr = i;
        }

        /* Remove duplicate sites */
        for(i=0;i<nsites;i++){
            if(!(sites[i].coord.x==sites[i+1].coord.x&&
                 sites[i].coord.y==sites[i+1].coord.y)){
                sites[index].coord.x=sites[i].coord.x;
                sites[index].coord.y=sites[i].coord.y;
                sites[index].label=sites[i].label;
                sites[index].sitenbr=index;
                index++;
            }
        }
        nsites=index;

        if(display_detail) printf("\t# of Connected Components (LABELnbr): %d\n", LABELnbr);
        if(display_detail) printf("\t# of Sites (nsites): %d\n", nsites);

        if(display_json) printf("\"numOfCC\":\"%d\",",LABELnbr);
        if(display_json) printf("\"numOfSites\":\"%d\",",nsites);
        metadata->numOfCC = LABELnbr;
        metadata->numOfSites = nsites;
    }  
  
    /*
      bf_edgelab_smpl
        Label the left edge while tracing the boundary line.
        Set sites during border tracking.
        Remove noise from sites using the length of the boundary as a criterion.
    */

    void bf_edgelab_smpl(ImageData *imgd, Coordinate x0, Coordinate y0,
                         Label ln)
    {
        Coordinate x1,y1,xn,yn;
        int tmp_nsites,total=1;
        Vector d,dold=Right;

        if(lab_get(x0,y0) == NOLABEL){
            xn = x0;
            yn = y0;
            d = next_point(imgd,&xn,&yn,&dold,DownLeft,ln);
            /* At this point d contains the initial value of the search in the next black pixel */

            if(vector_equal(d,Center)){
                bpxset(x0,y0,ln);
                return;
            }
            tmp_nsites = nsites;

            /* Sampling the first point */
            sites[nsites].coord.x = (float)x0;
            sites[nsites].coord.y = (float)y0;
            sites[nsites].label = ln;
            sites[nsites].refcnt = 0;
            nsites++;
            if(nsites%SITE_BOX==0)
                sites=(struct Site *)realloc(sites,(nsites+SITE_BOX)*sizeof*sites);
            sample_pix++;
    
            x1 = xn;
            y1 = yn;

            do{
                if(total%sample_rate==0){ 
                    sites[nsites].coord.x = (float)xn;
                    sites[nsites].coord.y = (float)yn;
                    sites[nsites].label = ln;
                    sites[nsites].refcnt = 0;
                    /* sites[nsites].sitenbr set after removing duplicate sites */
                    nsites++;
                    if(nsites%SITE_BOX==0)
                        sites=(struct Site *)realloc(sites,(nsites+SITE_BOX)*sizeof*sites);
                    sample_pix++;
                }
                total++;
                d = next_point(imgd,&xn,&yn,&dold,d,ln);
            }while(xn-dold.x!=x0||yn-dold.y!=y0||xn!=x1||yn!=y1);
            if((total-1)<=noise_max){ /* total-1 is the perimeter */
                nsites = tmp_nsites;
            }
        }
    }

    /*
      next_point
        A function that traces a boundary line and returns a vector to the next destination.
        Since xn and yn pass addresses, they are changed together.
    */
    Vector next_point(ImageData *imgd,
                      Coordinate *pxn, Coordinate *pyn,
                      Vector *pdold, Vector d, Label ln)
    {
        int z;

        for(z=0;z<7;z++){
            if(bit_get(imgd,(*pxn)+d.x,(*pyn)+d.y)==BLACK){
                edge_lab(*pdold,d,*pxn,*pyn,ln);
                *pdold = d;
                /* At this point d is the direction in which the direction goes */
                /* In the next stage this value is from which direction it came from */
                *pxn = (*pxn)+d.x;
                *pyn = (*pyn)+d.y;
                return(first_d(d));
            }
            d = rot_d(d);
        }
        return(Center);
    }

    /*
      first_d
      In the next pixel, a function that returns vector type as to where to start searching for black pixels from, which is the initial value of d of the next pixel.
    */

    Vector first_d(Vector d)
    {
        if(vector_equal(d,Right)||vector_equal(d,UpRight))
            return(DownRight);
        else if(vector_equal(d,Up)||vector_equal(d,UpLeft))
            return(UpRight);
        else if(vector_equal(d,Left)||vector_equal(d,DownLeft))
            return(UpLeft);
        else 
            return(DownLeft);
    }

    /*
      rot_d
      A function that turns d counterclockwise.
    */

    Vector rot_d(Vector d)
    {
        if(vector_equal(d,Right))
            return(UpRight);
        else if(vector_equal(d,UpRight))
            return(Up);
        else if(vector_equal(d,Up))
            return(UpLeft);
        else if(vector_equal(d,UpLeft))
            return(Left);
        else if(vector_equal(d,Left))
            return(DownLeft);
        else if(vector_equal(d,DownLeft))
            return(Down);
        else if(vector_equal(d,Down))
            return(DownRight);
        else
            return(Right);
    }

    /*
      edge_lab
      A function that performs labeling by a combination of 
      dold (where you came to the current pixel) and 
      d (which pixel you want to go to next).
    */
    void edge_lab(Vector dold, Vector d, Coordinate i, Coordinate j, Label ln)
    {
        if(vector_equal(dold,Right)){
        }
        else if(vector_equal(dold,UpRight)){
            if(vector_equal(d,DownLeft)){
                lab_set(i,j,ln);
                bpxset(i,j,ln);
            }
        }
        else if(vector_equal(dold,Up)){
            if(vector_equal(d,DownLeft)||vector_equal(d,Down)){
                lab_set(i,j,ln);
                bpxset(i,j,ln);
            }
        }
        else if(vector_equal(dold,UpLeft)){
            if(vector_equal(d,DownLeft)||vector_equal(d,Down)||
               vector_equal(d,DownRight)){
                lab_set(i,j,ln);
                bpxset(i,j,ln);
            }
        }
        else if(vector_equal(dold,Left)){
            if(vector_equal(d,DownLeft)||vector_equal(d,Down)||
               vector_equal(d,DownRight)||vector_equal(d,Right)){
                lab_set(i,j,ln);
                bpxset(i,j,ln);
            }
        }
        else if(vector_equal(dold,DownLeft)){
            if(vector_equal(d,DownLeft)||vector_equal(d,Down)||
               vector_equal(d,DownRight)||vector_equal(d,Right)||
               vector_equal(d,UpRight)){
                lab_set(i,j,ln);
                bpxset(i,j,ln);
            }
        }
        else if(vector_equal(dold,Down)){
            if(vector_equal(d,DownLeft)||vector_equal(d,Down)||
               vector_equal(d,DownRight)||vector_equal(d,Right)||
               vector_equal(d,UpRight)||vector_equal(d,Up)){
                lab_set(i,j,ln);
                bpxset(i,j,ln);
            }
        }
        else if(vector_equal(dold,DownRight)){
            if(vector_equal(d,DownLeft)||vector_equal(d,Down)||
               vector_equal(d,DownRight)||vector_equal(d,Right)||
               vector_equal(d,UpRight)||vector_equal(d,Up)||
               vector_equal(d,UpLeft)){
                lab_set(i,j,ln);
                bpxset(i,j,ln);
            }
        }
    }

    /*
      bpxset
      A function that sets x, y coordinates and label of the pixel 
      in bpx of type BlackPixel.
    */
    void bpxset(Coordinate i, Coordinate j, Label ln)
    {
        static unsigned int current_size=INITPIXEL;

        bpx[BPnbr].xax = i;
        bpx[BPnbr].yax = j;
        bpx[BPnbr].label=ln;
        BPnbr++;
        if(BPnbr>=current_size){
            bpx=(BlackPixel *)myrealloc(bpx,current_size,INCPIXEL,sizeof(BlackPixel));
            current_size+=INCPIXEL;
        }
    }
  
    /* sort sites on y, then x, coord */
    int scomp(const void *s1, const void *s2)
    {
        float s1x = ((struct Point *)s1)->x;
        float s1y = ((struct Point *)s1)->y;
        float s2x = ((struct Point *)s2)->x;
        float s2y = ((struct Point *)s2)->y;

        if(s1y < s2y) return(-1);
        if(s1y > s2y) return(1);
        if(s1x < s2x) return(-1);
        if(s1x > s2x) return(1);
        return(0);
    }
}
