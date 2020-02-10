#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "defs.h"
#include "const.h"
#include "function.h"
#include <limits.h>


namespace voronoi{
#define LINE_C  192 // blue color in range 0-255
#define WIDTH   5

    BlackPixel  *bpx;       /* Coordinates of black pixels and their labels */
    Shape       *shape;
    Zone        *zone;
    Neighborhood    *neighbor;  /* Characteristic quantity between adjacent connected components */
    LineSegment *lineseg;   /* Coordinates and labels of start and end points */
    HashTable   *hashtable[M1+M2];
    /* Hash table for labels of adjacent connected components */
    EndPoint    *endp;      /* End point of line segment */

    NumPixel    BPnbr;      /* Number of black pixels */
    Label           LABELnbr;   /* Number of connected components */
    unsigned int    NEIGHnbr;   /* Number of adjacent connected component sets */
    unsigned int    LINEnbr;    /* Number of line segments before removal Voronoi side */
    
    unsigned int    Enbr;       /* Number of connected component sets from which Voronoi sides are removed */
    
    long        SiteMax;    /* Maximum number of Voronoi points */

    int     noise_max = NOISE_MAX;     /* Number of pixels of connected component to remove */
    int     sample_rate = SAMPLE_RATE; /* Sampling with boundary tracking */
                       /* Percentage */
    float       freq_rate = FREQ_RATE;
    int             Ta = Ta_CONST;
    int             Ts = Ts_CONST;
    unsigned int    sample_pix; /* Pictures obtained by sampling */
    /* A prime number */
    unsigned int    point_edge; /* Point Voronoi number of sides */
    unsigned int    edge_nbr;   /* Area after removal Voronoi side */
    /* Number of line segments */
    int             *area;       /* Label in the area of ​​the connected component attached */

    // Modification by Faisal Shafait
    // keep track of noise components to remove them
    // from the output image
    bool *noise_comp;
    unsigned int nconcomp_inc=50;
    unsigned int nconcomp_size=0;
    // End of Modification
    
#ifdef TIME
    float    b_s_time=0;
    float    v_time=0;
    float    e_time=0;
    float    o_time=0;
    clock_t     start, end;
#endif /* TIME */

    float   xmin, xmax, ymin, ymax, deltax, deltay;

    struct Site     *sites;
    int         nsites;
    int         siteidx;
    int         sqrt_nsites;
    int         nvertices;
    struct Freelist     sfl;
    struct Site     *bottomsite;

    int         nedges;
    struct Freelist     efl;

    struct Freelist hfl;
    struct  Halfedge    *ELleftend, *ELrightend;
    int             ELhashsize;
    struct  Halfedge    **ELhash;

    int             PQhashsize;
    struct  Halfedge    *PQhash;
    int             PQcount;
    int             PQmin;

    /* ÄÉ²Ãµ¡Ç½ÍÑ */
    int    smwind = SMWIND;

    /* ÄÉ²ÃÊ¬ */
    char     output_points = NO;
    char     output_pvor = NO;
    char     output_avor = NO;
    char     display_parameters = NO;
    char     display_detail = NO;
    char     display_json = NO;

    float max ( float a, float b ) { return a > b ? a : b; }
    float min ( float a, float b ) { return a < b ? a : b; }

    void printProgress (double percentage)
    {
        int val = (int) (percentage * 100);
        int lpad = (int) (percentage * PBWIDTH);
        int rpad = PBWIDTH - lpad;
        printf ("\r\b%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
        fflush (stdout);
    }

    void voronoi_pageseg(LineSegment **mlineseg, 
                         unsigned int *nlines,
                         ImageData *imgd1,
                         Shape **mcc,
                         unsigned int *nccs,
                         MetaData *metadata) {

        bool DEBUG = false;

        point_edge = 0;
        edge_nbr = 0;

        BPnbr = LABELnbr = NEIGHnbr = LINEnbr = Enbr = SiteMax = 0;

        /* displaying parameters */
        if(display_parameters == YES)
            dparam();

        /* Set 1 pixels surrounding image to 0 */
        frame(imgd1,1,0);

        /* ¹õ²èÁÇbpx ¤ÎÎÎ°è³ÎÊÝ */
        bpx=(BlackPixel *)myalloc(sizeof(BlackPixel)* INITPIXEL);

        shape=(Shape * )myalloc(sizeof(Shape)* INITPIXEL);

        /* Site ·¿sites ¤ÎÎÎ°è³ÎÊÝ */
        sites = (struct Site *) myalloc(SITE_BOX*sizeof *sites);
    
        /* ÆþÎÏ²èÁü¤òSite ·¿¤ËÊÑ´¹ */
    
        if(display_detail) fprintf(stderr,"Transforming Image to Site...\n");
#ifdef TIME
        start = clock();
#endif
        img_to_site(imgd1,metadata);
#ifdef TIME
        end = clock();
        b_s_time = (float)((end-start)/((float)CLOCKS_PER_SEC));
#endif
        if(display_detail) fprintf(stderr,"done\n");

        /* area[ln] */
        area=(int *)myalloc(sizeof(int)*LABELnbr);

        /* area[ln] ¤ÎÃÍ¤ò½é´ü²½ */
        for(int i=0;i<LABELnbr;i++) area[i]=0;

        /* area[ln], set the value */
        for(int i=0;i<BPnbr;i++) area[bpx[i].label]++;

        for(int i=0;i<BPnbr;i++){
            shape[bpx[i].label].x_min=10000;
            shape[bpx[i].label].x_max=0;
            shape[bpx[i].label].y_min=10000;
            shape[bpx[i].label].y_max=0;
            //shape[bpx[i].label].conf={0.0};
            shape[bpx[i].label].conf_idx=0;
        }

        /* 
        shape [ [label:0, x_min:?, x_max:?, y_min:?, y_max:?],
                [label:1, x_min:?, x_max:?, y_min:?, y_max:?],
                [label:2, x_min:?, x_max:?, y_min:?, y_max:?],
                ...
                [label:n, x_min:?, x_max:?, y_min:?, y_max:?]]
        */
        
        for(int i=0;i<BPnbr;i++){
            if(shape[bpx[i].label].x_min > bpx[i].xax) shape[bpx[i].label].x_min=bpx[i].xax;
            if(shape[bpx[i].label].x_max < bpx[i].xax) shape[bpx[i].label].x_max=bpx[i].xax;
            if(shape[bpx[i].label].y_min > bpx[i].yax) shape[bpx[i].label].y_min=bpx[i].yax;
            if(shape[bpx[i].label].y_max < bpx[i].yax) shape[bpx[i].label].y_max=bpx[i].yax;
        }
        /*
        for(i=0;i<LABELnbr;i++){
            printf("\tx_min:%d x_max:%d y_min:%d y_max:%d\n",shape[i].x_min,shape[i].x_max,shape[i].y_min,shape[i].y_max);
        }
        */
        
        /* bpx ¤ÎÎÎ°è²òÊü */
        //        free(bpx);
    
        /* ÎÙÀÜÏ¢·ëÀ®Ê¬´Ö¤ÎÆÃÄ§ÎÌneighbor ¤ÎÎÎ°è³ÎÊÝ */
        neighbor = (Neighborhood *)myalloc(sizeof(Neighborhood)* INITNEIGHBOR);

        /* ÀþÊ¬lineseg ¤ÎÎÎ°è³ÎÊÝ */
        lineseg = (LineSegment *)myalloc(sizeof(LineSegment)* INITLINE);

        /* ¥Ï¥Ã¥·¥åÉ½¤ò½é´ü²½
           initialization of hash tables */
        init_hash();

        /* ¥¨¥ê¥¢Voronoi ¿ÞºîÀ® 
           constructing the area Voronoi diagram */
    
        if(display_detail) fprintf(stderr,"Constructing area Voronoi diagram...\n");
#ifdef TIME
        start = clock();
#endif
        voronoi(imgd1->imax, imgd1->jmax);
#ifdef TIME
        end = clock();
        v_time = (float)((end-start)/((float)CLOCKS_PER_SEC));
#endif

        if(display_detail) fprintf(stderr,"done\n");

        /* Debugging purpose. Mike */
        /*
        for(i=0;i<LABELnbr;i++){
            fprintf(stderr,"\t%d\n",area[i]);
        }
        */
        
        

        /* Allocate space of end point of Voronoi Edge */
        SiteMax+=1;
        endp = (EndPoint *)myalloc(sizeof(EndPoint) * SiteMax);
    
        /* Voronoi edge removal */
        if(display_detail) fprintf(stderr,"Erasing Voronoi edges...");
#ifdef TIME
        start = clock();
#endif
        erase(metadata);

#ifdef TIME
        end = clock();
        e_time = (float)((end-start)/((float)CLOCKS_PER_SEC));
#endif
        if(display_detail) fprintf(stderr,"done\n");


        /* neighbor ¤ÎÎÎ°è²òÊü */
        free(neighbor);
        
        /* ¥Ü¥í¥Î¥¤ÊÕ½ÐÎÏ */
#ifdef TIME
        start = clock();
#endif

        *nlines = LINEnbr;
        *mlineseg = (LineSegment *)malloc(LINEnbr*sizeof(LineSegment));
        for(int i=0;i<LINEnbr;i++) {
            (*mlineseg)[i] = lineseg[i];
            if(lineseg[i].yn == OUTPUT &&
               (lineseg[i].xs != lineseg[i].xe
                || lineseg[i].ys != lineseg[i].ye)) {
                edge_nbr++;
            }
        }

        /* To draw centroid of cc */
        
        *nccs = LABELnbr;
        *mcc = (Shape *)malloc(LABELnbr*sizeof(Shape));
        for(int i=0 ; i<LABELnbr ; i++)
        {
            (*mcc)[i] = shape[i];
        }
        
        if(display_detail) printf("\t# of Neighbors (NEIGHnbr): %d\n",NEIGHnbr);
        if(display_detail) printf("\t# of Lines (LINEnbr): %d\n",LINEnbr);
        if(display_detail) printf("\tArea after removal Voronoi side (edge_nbr): %d\n",edge_nbr);


#ifdef TIME
        end = clock();
        o_time = (float)((end-start)/((float)CLOCKS_PER_SEC));
#endif
        
        //write_image(img_mask_filename, &(img_mask->image), 1, 1);

        /* ÎÎ°è²òÊü */
        free(area);
        free(sites);
        free(lineseg);
        
        free(endp);
        freelist_destroy(&hfl);
        freelist_destroy(&efl);
        freelist_destroy(&sfl);

        if(display_detail) printf("Done.\n");
    }

    void set_param(int nm, int sr, float fr, int ta){
        if(nm>=0)
            noise_max = nm;
        if(sr>=0)
            sample_rate = sr;
        if(fr>=0)
            freq_rate = FREQ_RATE;
        if(ta>=0)
            Ta = ta;
    }

    void voronoi_colorseg(ImageData *out_img,
                          ImageData *in_img,
                          bool remove_noise) {
    
        unsigned int nlines=0;
        LineSegment  *mlineseg;
        voronoi_pageseg(&mlineseg,&nlines,in_img,NULL,NULL,NULL);

        /* setting image size */
        out_img->imax=in_img->imax;
        out_img->jmax=in_img->jmax;
        if((out_img->image=(char *)myalloc(in_img->imax*in_img->jmax))==NULL){
            fprintf(stderr,"voronoi_colorseg: not enough memory for image\n");
            exit(1);
        }
        bool noimage = false;
        bit_to_byte(in_img,out_img,noimage);

        if(remove_noise){
            for(int i=0;i<BPnbr; i++){
                int index = bpx[i].xax+(bpx[i].yax*out_img->imax);
                if(noise_comp[bpx[i].label] && index<out_img->imax*out_img->jmax)
                    out_img->image[index] = WHITE;
            }
        }

        for(int i=0;i<nlines;i++){
            if(mlineseg[i].yn == OUTPUT &&
               (mlineseg[i].xs != mlineseg[i].xe
                || mlineseg[i].ys != mlineseg[i].ye)) {
                draw_line(out_img, mlineseg[i].xs, mlineseg[i].ys, 
                          mlineseg[i].xe, mlineseg[i].ye, LINE_C, WIDTH);
                //             fprintf(stderr,"%d %d %d %d\n",
                //          mlineseg[i].xs,mlineseg[i].xe,
                //          mlineseg[i].ys,mlineseg[i].ye);
            }
        }
        free(bpx);
        free(shape);
        free(noise_comp);
        free(mlineseg);
    }
}