#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "coo.h"
#include "csr.h"

using namespace std;

sparse_coo coo;
sparse_csr csr;

//convert coo to csr 

int main()
{
    init(coo,100);
    insert(0,2,1,coo);
    insert(1,3,3,coo);
    insert(2,4,10,coo);
    printcoo(coo);

    int *rp;
    rp=(int *)malloc((coo.len+1) * sizeof(int));
    int count=0;

    for(int i=0;i<coo.len;i++){
        rp[coo.row_index[i]+1]++;
        if(rp[coo.row_index[i]+1]==1){
            count++;
        }
    }
    for(int i=0;i<count;i++){
        rp[i+1]+=rp[i];
    }
    

    initcsr(csr, coo.len, rp, coo.col_index, coo.val, count+1);
    printcsr(csr);


    return 0;
}