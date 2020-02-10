# coding: utf-8

from diqa import estimate_skew as skew, estimate_contrast as contrast, estimate_rangeeffect as rangeeffect, estimate_bleedthrough as bleedthrough
from multiprocessing.dummy import Pool as ThreadPool

def run_one(line):

    try:
        x = line[0:len(line)-1]
        sk = skew(x)
        con = contrast(x)
        ra = rangeeffect(x)
        bl = bleedthrough(x)
        
        return (",".join([x, str(sk), str(con), str(ra), str(bl)]))
        
    except Exception as e:
        return (",".join([line, str(e)]))
    

# function to be mapped over
def calculateParallel(lines, threads=2):
    pool = ThreadPool(threads)
    rslt = pool.map(run_one, lines)
    pool.close()
    pool.join()
    return rslt

if __name__ == "__main__":
    filepath = 'filelist.txt'
    lines = []
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            lines.append(line)
            line = fp.readline()

    rslt = calculateParallel(lines, 8)
    
    with open('diqa_rslt.csv', 'w') as f:
        for item in rslt:
            f.write("%s\n" % item)
