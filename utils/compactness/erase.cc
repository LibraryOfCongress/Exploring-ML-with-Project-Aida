/*
  $Date: 1999/10/18 08:59:27 $
  $Revion: $
  $Author: kise $
  erase.c
  Voronoi edge removal program

   
    When the distance between connected components corresponding to 
    one Voronoi side is equal to or less than the threshold value, 
    the Voronoi side is removed.
    As a result, Voronoi edges with simple endpoints can also be removed.

    endp : The entry of this array is the number of the end point of the Voronoi side
    The first value (line) is the number of line segments (Voronoi edges) having this Voronoi point as an end point
    The value (line) of the area reserved after that is the number of the line segment (Voronoi side) having this Voronoi point as the end point

    lineseg : 
    Remember the number of the end point (Voronoi point) of the Voronoi side, 
    the coordinates of the start and end points of the Voronoi side, 
    and the output permission information
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "const.h"
#include "defs.h"
#include "extern.h"
#include "function.h"

namespace voronoi{
    extern int    *area;

    float         Td2_coef=1.0, Td2_coef2=1.0;
    float		  Td2=0.0;	/* Coefficient of discriminant formula */
    unsigned int  Td1=0;		/* Distance threshold */

    unsigned int  Dmax,Amax;
    unsigned int MFS; /* Most frequent CC size */
    float size_std;
    float size_mean;
    float MFA;
    float angle_dev;
    float Q1, Q3;
    int        *angle;
    

    int start_pos (int pos)
    {
        int cpos = pos - smwind;
        if (cpos < 0) {
            return 0;
        }
        else {
            return cpos;
        }
    }

    int end_pos (int pos)
    {
        int cpos = pos + smwind;
        if (cpos >= Dmax){
            return Dmax-1;
        }
        else {
            return cpos;
        }
    }

    unsigned int Dh_ave(unsigned int *Dh, int pos)
    {
        int i;
        unsigned int ave=0;

        int start = start_pos(pos);
        int end = end_pos(pos);

        for(i=start;i<=end;i++){
            ave=ave+Dh[i];
        }
        return ( (unsigned int) (ave / (end - start + 1)) );
    }
    unsigned int Ah_ave(unsigned int *Ah, int pos)
    {
        int i;
        unsigned int ave=0;

        int start = start_pos(pos);
        int end = end_pos(pos);

        for(i=start;i<=end;i++){
            ave=ave+Ah[i];
        }
        return ( (unsigned int) (ave / (end - start + 1)) );
    }

    /*
      A function to create a histogram of the distance, the ratio of black pixels (difference), the average black run length difference
    */

    int mostFrequent(int arr[], int n) 
    { 
        int fArray[n];
        int new_n = 0;
        // Sort the array 
        //sort(arr, arr + n); 
        //qsort(arr, n, sizeof(int),compare); 
        for(int i=0 ; i<n ; i++)
        {
            if(NOISE_MAX<arr[i] and (Q1-1.5*(Q3-Q1)<arr[i] and arr[i]<Q3+1.5*(Q3-Q1)))
            {
                fArray[i] = arr[i];    
                new_n++;
            }
        }
        qsort(fArray, new_n, sizeof(int),compare); 
        /*
        // find the max frequency using linear traversal 
        int max_count = 1, res = arr[0], curr_count = 1; 
        for (int i = 1; i < n; i++) { 
            if (arr[i] == arr[i - 1]) 
                curr_count++; 
            else { 
                if (curr_count > max_count) { 
                    max_count = curr_count; 
                    res = arr[i - 1]; 
                } 
                curr_count = 1; 
            } 
        } 
      
        // If last element is most frequent 
        if (curr_count > max_count) 
        { 
            max_count = curr_count; 
            res = arr[n - 1]; 
        } 
        */
        return fArray[median(fArray, 0, new_n)]; 
    } 

    // A comparator function used by qsort 
    int compare(const void * a, const void * b) 
    { 
        return ( *(int*)a - *(int*)b ); 
    } 
    
    float calculateMean(int data[], int n)
    {
        int sum = 0, mean;
        int myN = 0;
        
        for(int i=0 ; i<n ; i++)
        {
            //if(data[i]<1031.75 and data[i]>30){
            if(NOISE_MAX<data[i] and (Q1-1.5*(Q3-Q1)<data[i] and data[i]<Q3+1.5*(Q3-Q1)))
            {
                sum += data[i];
                myN++;
            }
        }
        mean = sum/myN;

        return mean;
    }

    float calculateSD(int data[], int n)
    {
        int sum = 0;
        float mean, standardDeviation = 0.0;

        for(int i=0; i<n; i++)
        {
            sum += data[i];
        }
        mean = sum/(float)n;

        for(int i=0; i<n; i++)
            standardDeviation += pow(data[i] - mean, 2);

        return sqrt(standardDeviation/n);
    }

    // Function to give index of the median 
    int median(int* arr, int l, int r) 
    { 
        int n = r - l + 1; 
        n = (n + 1) / 2 - 1; 
        return n + l; 
    } 
      
    // Function to calculate IQR 
    void IQR(int arr[], int n, float* Q1, float* Q3) 
    { 
        // copy arr
        int arr_copy[n];
        for(int i=0 ; i<n ; i++)
        {
            arr_copy[i] = arr[i];
        }
        qsort(arr_copy, n, sizeof(int),compare); 

        // Standardize (Found that standardization happens to reduce variability, so this is commented out.)
        /*
        float max = arr[n-1];
        float min = arr[0];

        for(int i=0 ; i<n ; i++)
        {
            arr[i] = (arr[i]-min)/(max-min);    
        }
        */
      
        // Index of median of entire data 
        int mid_index = median(arr_copy, 0, n); 
      
        // Median of first half 
        *Q1 = arr_copy[median(arr_copy, 0, mid_index)]; 
      
        // Median of second half 
        *Q3 = arr_copy[median(arr_copy, mid_index + 1, n)]; 
    } 

    void hist(MetaData * metadata)
    {
        FILE        *dist_ofp;
        FILE        *angle_ofp;

        int i,j;
        unsigned int *Dh, *Dh_ref, *Ah, *Ah_ref, *Anh, *Anh_ref, max1, max2;
        float freq,freq1,freq2;

        Dmax = 0;
        Amax = 0;

    

        /* Find the maximum value of the distance, the ratio of black pixels (difference), the difference of average black run length */
        for(i=0;i<NEIGHnbr;i++) {
            if(Dmax < neighbor[i].dist)
                Dmax = (unsigned int)(neighbor[i].dist+0.5);
        }
        Dmax++;
        for(i=0;i<LABELnbr;i++) {
            if(Amax < area[i])
                Amax = (unsigned int)(area[i]);
        }
        Amax++;
        /* Allocate memory for the distance histogram array */
        Dh = (unsigned int *)myalloc(sizeof(unsigned int)* Dmax);
        Dh_ref = (unsigned int *)myalloc(sizeof(unsigned int)* Dmax);
        Ah = (unsigned int *)myalloc(sizeof(unsigned int)* Amax);
        Ah_ref = (unsigned int *)myalloc(sizeof(unsigned int)* Amax);
        angle = (int *)myalloc(sizeof(int)*NEIGHnbr);

        /* Initialize array */
        for(i=0;i<Dmax;i++) {
            init_u_int(&Dh[i]);
            init_u_int(&Dh_ref[i]);
        }
        for(i=0;i<Amax;i++) {
            init_u_int(&Ah[i]);
            init_u_int(&Ah_ref[i]);
        }
        for(i=0;i<NEIGHnbr;i++){
            angle[i] = (int)(neighbor[i].angle*100);
        }

        IQR(area,LABELnbr,&Q1,&Q3);
        if(display_detail) printf("\n\tIQR: %.1lf (Q1:%.1f,Q3:%.1f)\n",Q3-Q1,Q1,Q3);
        MFS = mostFrequent(area, LABELnbr);
        if(display_detail) printf("\tMost Frequent Size: %d\n",MFS);
        size_mean = calculateMean(area, LABELnbr);
        if(display_detail) printf("\tMean Size: %1.f\n",size_mean);
        size_std = calculateSD(area, LABELnbr);
        if(display_detail) printf("\tSTD Size: %.1f\n", size_std);
        MFA = mostFrequent(angle,NEIGHnbr)/float(100);
        if(display_detail) printf("\tMost Frequent Angle: %.2lf\n",MFA);
        
        if(display_json) printf("\"IQR\":\"%.1f\",",Q3-Q1);
        if(display_json) printf("\"Q1\":\"%.1f\",",Q1);
        if(display_json) printf("\"Q3\":\"%.1f\",",Q3);
        if(display_json) printf("\"sizeMean\":\"%.1f\",",size_mean);
        if(display_json) printf("\"sizeStd\":\"%.1f\",",size_std);
        if(display_json) printf("\"angleMode\":\"%.2lf\",",MFA);
        metadata->IQR = Q3-Q1;
        metadata->Q1  = Q1;
        metadata->Q3  = Q3;
        metadata->sizeMean  = size_mean;
        metadata->sizeStd   = size_std;
        metadata->angleMode = MFA;

        /* Histogram creation */
        for(i=0;i<NEIGHnbr;i++) {
            Dh[(int)(neighbor[i].dist+0.5)]++;
        }
        for(i=0;i<LABELnbr;i++) {
            Ah[(int)(area[i])]++;
        }

        /* Smoothing histogram */
        for(i=0;i<Dmax;i++){
            Dh_ref[i]=Dh[i];
        }
        for(i=0;i<Dmax;i++){
            Dh[i]=Dh_ave(Dh_ref,i);
        }
        
        //MFA = mostFrequent(angle,1);
        //printf("\tMost Frequent Angle: %.2lf\n",MFA);

        /*
        for(i=0;i<Amax;i++){
            Ah_ref[i]=Ah[i];
        }
        for(i=0;i<Amax;i++){
            Ah[i]=Ah_ave(Ah_ref,i);
        }
        */

        /* Debug: Print areas */
        /*
        for(i=0;i<Amax;i++){
            printf("\t%d\n",Ah[i]);
        }
        */

        /* Examine the histogram */

        /* Decide the value of constant value Td 2 on distance */
        max1 = max2  = 0;
        for(i=1;i<Dmax-1;i++) {
	
            /* When i is the maximum point */
            if(Dh[i-1] < Dh[i] && Dh[i] > Dh[i+1]){
                if(Dh[max1] < Dh[i]) {
                    max2 = max1;
                    max1 = i;
                }
                else if(Dh[max2] < Dh[i])
                    max2 = i;
            }
            else if(Dh[i-1] == Dh[i] && Dh[i] > Dh[i+1]) {
                for(j=i-2;j>=0;j--) {
                    if(Dh[j] < Dh[i]) {
                        if(Dh[max1] < Dh[i]) {
                            max2 = max1;
                            max1 = i;
                        }
                        else if(Dh[max2] < Dh[i])
                            max2 = i;

                        break;
                    }
                    else if(Dh[j] > Dh[i])
                        break;
                }
            }
        }
        if(max1 > max2) {
            i = max2;
            max2 = max1;
            max1 = i;
        }
    
        /*
          linear interpolation between (i, Dh [i]) and (i + 1, Dh [i + 1])
        */
        freq=(float)Dh[max2]*freq_rate;
        for(i=max2 ; i<Dmax-1 ; i++) {
            freq1=(float)Dh[i];
            freq2=(float)Dh[i+1];
            if(freq1 >= freq && freq >= freq2){
                if(freq1 != freq2){
                    Td2=(freq1*(float)(i+1)-freq2*(float)i-freq)/(freq1-freq2);
                }
                else{
                    for(j=i+1;j<Dmax;j++){
                        if(Dh[j]!=freq){
                            Td2=(float)(i+j-1)/2.0;
                            /* In the case of parallel, the middle point */
                            break;
                        }
                    }
                }
                break;
            }
        }

        Td1 = max1;

        /* Mike - print out Dh for examination purpose */
        /*
        fprintf(stderr, "\nHistogram\n");
        for(i=0;i<Dmax;i++){
            fprintf(stderr, "%lu\n", Dh[i]);
        }
        fprintf(stderr, "\n");
        */

        if(display_detail)
        {
            fprintf(stderr,
                "\n\tdist\tmax1 %d  max2 %d : Td1 %d  Td2 %.1f  Ta %d\n",
                max1,max2,Td1,Td2,Ta);
        }
        if(display_json) printf("\"td1\":\"%d\",",Td1);
        if(display_json) printf("\"td2\":\"%.1f\",",Td2);
        metadata->td1 = Td1;
        metadata->td2 = Td2;

        if(Td2 == 0.0) {
            fprintf(stderr,"The value of coefficient Td2 is 0\n");
            exit(1);
        }

        /* Output distance histogram*/
        /*
        if((dist_ofp = fopen("dist_hist","w"))==NULL) {
            fprintf(stderr,"can't open output-file\n");
            exit(1);
        }
        for(i=0;i<Dmax;i++) {
            fprintf(dist_ofp,"%d\n",Dh[i]);
        }
        fclose(dist_ofp);
        */
        /* Output angle histogram */
        /*
        if((angle_ofp = fopen("angle_hist","w"))==NULL) {
            fprintf(stderr,"can't open output-file\n");
            exit(1);
        }
        for(i=0;i<NEIGHnbr;i++) {
            fprintf(angle_ofp,"%.2lf\n",neighbor[i].angle);
        }
        fclose(angle_ofp);
        */


        
        /* Space release */
        free(Dh);
        free(Dh_ref);
    }

    /*
      From the distance between the two connected components, 
      the difference in the number of black pixels, (the difference in the average black run length), 
      a function to determine whether the Voronoi side can be output for the time being
    */   
    int distinction(Label lab1, Label lab2, int j, int i)
    {
        float shape1,shape2,dshape,dist,dxy,xy1,xy2,n,n_test,td_test;
            
        /* Distance between connected components */
        dist = neighbor[j].dist;

        //if(dist <= Td1) return(NO_OUTPUT);	/* Do not output (remove) */
        /*
        shape1 = (float)((shape[lab1].x_max-shape[lab1].x_min)*(shape[lab1].y_max-shape[lab1].y_min));
        shape2 = (float)((shape[lab2].x_max-shape[lab2].x_min)*(shape[lab2].y_max-shape[lab2].y_min));
        */
        
        if((shape[lab1].x_max-shape[lab1].x_min) > (shape[lab1].y_max-shape[lab1].y_min)){
            shape1 = (float)(shape[lab1].x_max-shape[lab1].x_min)/(shape[lab1].y_max-shape[lab1].y_min);
        }
        else{
            shape1 = (float)(shape[lab1].y_max-shape[lab1].y_min)/(shape[lab1].x_max-shape[lab1].x_min);
        }
        if((shape[lab2].x_max-shape[lab2].x_min) > (shape[lab2].y_max-shape[lab2].y_min)){
            shape2 = (float)(shape[lab2].x_max-shape[lab2].x_min)/(shape[lab2].y_max-shape[lab2].y_min);
        }
        else{
            shape2 = (float)(shape[lab2].y_max-shape[lab2].y_min)/(shape[lab2].x_max-shape[lab2].x_min);
        }
        /*
        if(shape1 > shape2)
            dshape = shape1 / shape2;
        else
            dshape = shape2 / shape1;
        */
        if(shape1 > shape2)
            dshape = shape1;
        else
            dshape = shape2;
        

        /* Number of black pixels of two connected components */
        xy1 = (float)area[lab1];
        xy2 = (float)area[lab2];

        //printf("xy1:%.1f xy2:%.1f\n",xy1,xy2);
        
        if(xy1 > xy2)
            dxy = xy1 / xy2;
        else
            dxy = xy2 / xy1;

        /* Mike Voronoi */        
        // Version 2.
        // Only for components with SIMILAR (font) SIZE
        if(dxy<SIZE_SIM)// and dshape<1.5)
        {
            // Use min(xy1,xy2) for the exponential function in order to attenuate false-alarm (i.e., two componetns are not quite similar to each other.)
            if(xy1 < xy2)
                //Td2_coef = 2/(1+exp(-SHAPE_K*(xy1-size_mean)))+0.001;
                Td2_coef = 2/(1+exp(-(1/(float)size_std/2)*(xy1-size_mean)))+0.001;
            else
                //Td2_coef = 2/(1+exp(-SHAPE_K*(xy2-size_mean)))+0.001;
                Td2_coef = 2/(1+exp(-(1/(float)size_std/2)*(xy1-size_mean)))+0.001;
            // Cap range of Td2_coef from 1.0 to 2.0
            if (Td2_coef<1) Td2_coef = 1.0;
            /*
            if (Td2_coef>2) Td2_coef = 2.0;
            else if (Td2_coef<1) Td2_coef = 1.0;
            */
        }
        else{
            Td2_coef = 1.0;
        }

    
        
        
        n = dist / (Td2_coef*Td2) + dxy / (float)Ta + dshape / (float)Ts_CONST;
        //printf("\tlineseg[%d]..label1:%d label2: %d ... (%d,%d) (%d,%d)\n\t\tdist:%.2f\n\t\tdshape:%.1f (shape1:%.1f shape2:%.1f)\n\t\tdxy:%.1f (xy1:%.1f xy2:%.1f)\n\t\tTd2_coef:%.1f ... n:%.2f\n",i,lab1,lab2,shape[lab1].x_min,shape[lab1].y_min,shape[lab2].x_min,shape[lab2].y_min,dist,dshape,shape1,shape2,dxy,xy1,xy2,Td2_coef,n);
        
        /* Original Voronoi */
        //n = dist / Td2 + dxy / (float)Ta;
        //printf("\tlineseg[%d]..label1:%d label2: %d ... (%d,%d) (%d,%d)\n\t\tdist:%.2f\n\t\tdxy:%.1f (xy1:%.1f xy2:%.1f) ... n:%.2f\n",i,lab1,lab2,shape[lab1].x_min,shape[lab1].y_min,shape[lab2].x_min,shape[lab2].y_min,dist,dxy,xy1,xy2,n);
        
        /* Voronoi++ (Mike - 2019/01/13) */
        //if(xy1<MFS or xy2<MFS) return(NO_OUTPUT); 
        
        
        
       
        /*
        n_test = dist / Td2 + dxy / (float)Ta;
        
        td_test = Td2*((xy1+xy2)/2.0)/(float)MFS;
        angle_dev = exp(-(pow(neighbor[j].angle-MFA,2))/(2*pow(ANGLE_SIGMA,2)))*ANGLE_K;
        td_test = td_test+td_test*angle_dev;
        n = dist/(td_test-td_test*dxy/(float)Ta)+dxy/(float)Ta;
    */
        //printf("\tn:%.2lf dist:%.2lf Td2:%.2lf dxy:%.2lf Ta:%d\n",n,dist,Td2,dxy,Ta);
        //printf("%d\n",dist);
        //printf("lineseg[%d]..label1:%d label2: %d ... (%d,%d) (%d,%d)\n\t\tdist:%.2f\n\t\tTd2_new:%.1f angle_dev:%.2f\n\t\tdxy:%.1f (xy1:%.1f xy2:%.1f)\n\t\tn_old:%.2f ... n_new:%.2f\n",i,lab1,lab2,shape[lab1].x_min,shape[lab1].y_min,shape[lab2].x_min,shape[lab2].y_min,dist,td_test,angle_dev,dxy,xy1,xy2,n_test,n);
        //printf("\nDetail: Distance: %lf, ar: %lf, n_old: %lf, n_new: %lf", dist, dxy, n_test, n);
        
        // Version 3.
        /*
        shape[lab1].conf[shape[lab1].conf_idx++] = n;
        shape[lab2].conf[shape[lab2].conf_idx++] = n;
        lineseg[i].conf = n;
        */

        if(n >= CONF_LEVEL){
            return(OUTPUT);		/* Output (not removed) */
        }
        else{
            return(NO_OUTPUT);	/* Do not output (remove) */
        }
    }

    /* Function to eliminate Voronoi edges with simple endpoints */
    void erase_endp(int j)
    {
        EndPoint *point;

        while(j != FRAME) {
            endp[j].line--;
            /* When the number of Voronoi edges is 1 */
            if(endp[j].line == 1) {
                point = endp[j].next;
                /* Set the number of Voronoi edges to 0 */
                endp[j].line = 0;
                while(1) {

                    /* The other end point of the Voronoi side that is still to be output */
                    if(lineseg[point->line].yn == OUTPUT) {
                        if(j == lineseg[point->line].sp){
                            j = lineseg[point->line].ep;
                        }
                        else{
                            j = lineseg[point->line].sp;
                        }

                        /* Do not output */
                        lineseg[point->line].yn = NO_OUTPUT;
                        /* Remove the next point j if it can be removed */
                        break;
                    }
                    /* When I finished watching the list of Voronoi edges in full */
                    else if(point->next == NULL){

                        return;
                    }
                    /* Call up next */
                    else point = point->next;
                }
                continue;
            }
            else return;
        }
    }

    void erase_aux()
    {
        int i,j;
        EndPoint *point;

        /* Initialize endp */
        for(i=0;i<SiteMax;i++){
            init_int(&endp[i].line);
            endp[i].next = NULL;
        }

        /*
         * If the distance between connected components is greater than or equal to the threshold (THRESHOLD),
         * examine Voronoi edges 
         */
        for(i=0;i<LINEnbr;i++) {
            j = search(lineseg[i].lab1,lineseg[i].lab2);

            /* If Voronoi side is discriminant formula, if it is possible to output for the time being */
            if(distinction(lineseg[i].lab1,lineseg[i].lab2,j,i)
               == OUTPUT) {
                // Comment out below two if statements for version 3
                // When the start point is not the edge of the image 
                if(lineseg[i].sp != FRAME) {
                    point = (EndPoint *)myalloc(sizeof(EndPoint)* 1);
	    
                    // Connect allocated memory to endp []
                    point->next = endp[lineseg[i].sp].next;
                    endp[lineseg[i].sp].next = point;
                    point->line = i;
                    endp[lineseg[i].sp].line ++;
                }

                // When the end point is not the edge of the image
                if(lineseg[i].ep != FRAME) {
                    point = (EndPoint *)myalloc(sizeof(EndPoint)* 1);
	   
                    // Connect allocated memory to endp []
                    point->next = endp[lineseg[i].ep].next;
                    endp[lineseg[i].ep].next = point;
                    point->line = i;
                    endp[lineseg[i].ep].line ++;
                }
                
                lineseg[i].yn = OUTPUT; // Suppose you can output it for the time being 
            }

            /* Do not output */
            else {
                lineseg[i].yn = NO_OUTPUT;
            }
        }

        /* Version 3: Confidence-based re-processing */
        /*
        // Calculate IQR of confidence of edges surronding CC.
        for(i=0;i<LABELnbr;i++) 
        {
            int median_idx = median(shape[i].conf,0,shape[i].conf_idx);
            
            shape[i].conf_iqr = IQR(shape[i].conf,shape[i].conf_idx);
            
            //printf("\"Label\":\"%d\", \"IQR(Conf[])\":\"%.2f\"\n",i,shape[i].conf_iqr);
            for(j=0;j<shape[i].conf_idx;j++)
            {
                if(j==median_idx)
                {
                    shape[i].conf_median = shape[i].conf[j]; 
                }
                //printf("%.2f ",shape[i].conf[j]);
            }
            //printf("\n\"median(Conf[%d])\":\"%.2f\"\n",median_idx,shape[i].conf_median);
            
        }
        // Reconsider whether edge should be deleted or not based on the adjusted-IQA-based confidence.
        
        for(i=0;i<LINEnbr;i++) 
        {
            float max_iqr;
            float max_median;
            if(shape[lineseg[i].lab1].conf_iqr > shape[lineseg[i].lab2].conf_iqr){
                max_iqr = shape[lineseg[i].lab1].conf_iqr;   
                max_median = shape[lineseg[i].lab1].conf_median;
            }
            else{
                max_iqr = shape[lineseg[i].lab2].conf_iqr;   
                max_median = shape[lineseg[i].lab2].conf_median;
            }
            //printf("Line[%d] between Label[%d] and Label[%d]:\n",i,lineseg[i].lab1,lineseg[i].lab2);
            //printf("\tLine.conf %.2f -> conf_iqr %.2f\n",lineseg[i].conf, lineseg[i].conf+max_iqr);
            if(lineseg[i].conf>=max_median and lineseg[i].conf + max_iqr > CONF_LEVEL)
            {

                // When the start point is not the edge of the image 
                if(lineseg[i].sp != FRAME) {
                    point = (EndPoint *)myalloc(sizeof(EndPoint)* 1);
        
                    // Connect allocated memory to endp []
                    point->next = endp[lineseg[i].sp].next;
                    endp[lineseg[i].sp].next = point;
                    point->line = i;
                    endp[lineseg[i].sp].line ++;
                }

                // When the end point is not the edge of the image
                if(lineseg[i].ep != FRAME) {
                    point = (EndPoint *)myalloc(sizeof(EndPoint)* 1);
       
                    // Connect allocated memory to endp []
                    point->next = endp[lineseg[i].ep].next;
                    endp[lineseg[i].ep].next = point;
                    point->line = i;
                    endp[lineseg[i].ep].line ++;
                }
                lineseg[i].yn = OUTPUT;
            }
            else{
                lineseg[i].yn = NO_OUTPUT;   
            }
        }
        */
        // Version 3 end.

        
        /* End point removal */
        for(i=0;i<SiteMax;i++) {
            /* At the end point that belongs only to one Voronoi side */
            if(endp[i].line == 1) {
                point = endp[i].next;

                endp[i].line = 0;	/* Set the number of Voronoi edges to 0 */

                lineseg[point->line].yn = NO_OUTPUT; /* Do not output */
	    
                /* Find the other endpoint of the Voronoi side */
                if(i == lineseg[point->line].sp){
                    j = lineseg[point->line].ep;
                }
                else if(i == lineseg[point->line].ep) {
                    j = lineseg[point->line].sp;
                }
                else {
                    fprintf(stderr,"erase() error(1) !!\n");
                    exit(1);
                }
	    	    
                erase_endp(j);	/* Check if point j can be removed and remove it if possible */
            }
        }
        //erase_endp(j);
    }

    /* Voronoi edge elimination function */
    void erase(MetaData *metadata)
    {
        /* Calculate the histogram, calculate the threshold value of the discriminant */
        hist(metadata);
        erase_aux();
    }

    /* Initialize unsigned int type */
    void init_u_int(unsigned int *data)
    {
        *data = 0;
    }

    /* Initialize int type */
    void init_int(int *data)
    {
        *data = 0;
    }
}
