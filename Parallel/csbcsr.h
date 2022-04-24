#ifndef _CSBCSR_H
# define _CSBCSR_H

typedef struct csbcsr_describe{
    int block_num;
    int length;
    int *rows;
    int *cols;
    int *val;
} sparse_csbcsr;

void initcsbcsr(sparse_csbcsr & bcsr, int bn, int l, int *rs, int *cs, int *v){
    bcsr.block_num=bn;
    bcsr.length=l;
    bcsr.rows=(int *)malloc((bn) * sizeof(int));
    bcsr.cols=(int *)malloc((bn) * sizeof(int));
    bcsr.val=(int *)malloc((bn*l*l) * sizeof(int));

    for(int i=0;i<bn;i++){
        bcsr.rows[i]=rs[i];
        bcsr.cols[i]=cs[i];
    }
    for(int i=0;i<bn*l*l;i++){
        bcsr.val[i]=v[i];
    }
}

void printcsbcsr(const sparse_csbcsr & csr){
    printf("csbcsr rows:\n");
    for(int i=0;i<csr.block_num;i++){
        printf("%d ",csr.rows[i]);
    }
    printf("\n");

    printf("csbcsr cols:\n");
    for(int i=0;i<csr.block_num;i++){
        printf("%d ",csr.cols[i]);
    }
    printf("\n");

    printf("csbcsr val:\n");
    for(int i=0;i<csr.block_num*csr.length*csr.length;i++){
        printf("%d ",csr.val[i]);
    }
    printf("\n");
}

#endif