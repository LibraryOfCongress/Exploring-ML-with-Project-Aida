/*
   $Date: 1999/10/15 12:40:27 $
   $Revision: 1.1.1.1 $
   $Author: kise $
   main.c
   メインプログラム
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "defs.h"
#include "extern.h"
#include "const.h"
#include "function.h"
#include <sys/resource.h>

using namespace voronoi;
int main(int argc, char **argv) {
    FILE		*ofp,*ofp2,*ofp3;
    int 		i;
    int                 ifargc, ofargc, of2argc;
    ImageData		imgd1;
    MetaData        metadata;

    /* analysis of arguments */
    analyze_cline(argv,&ifargc);
   //neighbor = (Neighborhood *)myalloc(sizeof(Neighborhood)* INITNEIGHBOR);

    
    /* ファイルオープン
       opening the output file */
    if((ofp = fopen("./data/linesegments/line","w"))==NULL) {
	fprintf(stderr,"can't open line output-file\n");
	exit(1);
    }
    
    if((ofp2 = fopen("./data/ccs/cc","w"))==NULL) {
    fprintf(stderr,"can't open cc output-file\n");
    exit(1);
    }

    if((ofp3 = fopen("./data/metadata/metadata","w"))==NULL) {
    fprintf(stderr,"can't open metadata output-file\n");
    exit(1);
    }
    
    //strcpy(*metadata.filename,argv[ifargc]);
    metadata.filename = argv[ifargc];
    if(display_json) printf("{\"fileName\":\"%s\",",argv[ifargc]);
    
    /* reading the image data */
    read_image(argv[ifargc],&imgd1);
    metadata.width  = imgd1.imax;
    metadata.height = imgd1.jmax;

    unsigned int nlines=0;
    LineSegment	 *mlineseg;
    unsigned int nccs = 0;
    Shape        *mcc;
    voronoi_pageseg(&mlineseg,&nlines,&imgd1,&mcc,&nccs,&metadata);

    /* Write linesegments */
    /* 
    // Original
    for(i=0;i<nlines;i++) {
    	if(mlineseg[i].yn == OUTPUT &&
    	   (mlineseg[i].xs != mlineseg[i].xe
    	    || mlineseg[i].ys != mlineseg[i].ye)) {
    	    fprintf(ofp,"%d %d %d %d\n",
    		    mlineseg[i].xs,mlineseg[i].xe,
    		    mlineseg[i].ys,mlineseg[i].ye);
    	}
    }
    */

    // For Second paper
    for(i=0;i<nlines;i++) {
      if(mlineseg[i].yn == OUTPUT &&
         (mlineseg[i].xs != mlineseg[i].xe
          || mlineseg[i].ys != mlineseg[i].ye)) {
          fprintf(ofp,"%d %d %d %d\n",
            mlineseg[i].xs,mlineseg[i].xe,
            mlineseg[i].ys,mlineseg[i].ye,
            mcc[mlineseg[i].lab1].x_min,
            mcc[mlineseg[i].lab1].x_max
            );
      }
    }

    /* Write conneted components */
    for(i=0;i<nccs;i++) {
        fprintf(ofp2,"%d %d\n",
            (mcc[i].x_min+mcc[i].x_max)/2,
            (mcc[i].y_min+mcc[i].y_max)/2);
    }

    /* Build JSON */
    fprintf(ofp3,"{\"width\":\"%d\",\"height\":\"%d\",\"numOfCC\":\"%d\",\"numOfSites\":\"%d\",\"IQR\":\"%.0f\",\"Q1\":\"%.0f\",\"Q3\":\"%.0f\",\"sizeMean\":\"%.0f\",\"sizeStd\":\"%.0f\",\"angleMode\":\"%.1f\",\"td1\":\"%d\",\"td2\":\"%.0f\"}",
                   metadata.width,
                   metadata.height,
                   metadata.numOfCC,
                   metadata.numOfSites,
                   metadata.IQR,
                   metadata.Q1,
                   metadata.Q3,
                   metadata.sizeMean,
                   metadata.sizeStd,
                   metadata.angleMode,
                   metadata.td1,
                   metadata.td2
                   );
/*
    printf("METADATA: %s\n",metadata.filename);
    printf("METADATA: %d\n",metadata.width);
    printf("METADATA: %d\n",metadata.height);
    printf("METADATA: %d\n",metadata.numOfCC);
    printf("METADATA: %d\n",metadata.numOfSites);
    printf("METADATA: %.0f\n",metadata.IQR);
    printf("METADATA: %.0f\n",metadata.Q1);
    printf("METADATA: %.0f\n",metadata.Q3);
    printf("METADATA: %.0f\n",metadata.sizeMean);
    printf("METADATA: %.0f\n",metadata.sizeStd);
    printf("METADATA: %.1f\n",metadata.angleMode);
    printf("METADATA: %d\n",metadata.td1);
    printf("METADATA: %.0f\n",metadata.td2);
*/

    /* ファイルクローズ */
    fclose(ofp);
    fclose(ofp2);
    fclose(ofp3);
    free(bpx);
    free(noise_comp);
    free(imgd1.image);
    if(!nlines)
        free(mlineseg);
    if(!nccs)
        free(mcc);
#ifdef TIME
    dtime();
#endif
}
