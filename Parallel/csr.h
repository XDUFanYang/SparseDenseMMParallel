#ifndef _CSR_H
# define _CSR_H

typedef struct csr_describe{
    int* row_ptr;
    int* col_index;
    int* val;
    int len;
    int rpl;
} sparse_csr;

void initcsr(sparse_csr & csr, int l, int *rp, int *ci, int *v, int rpl){
    csr.len=l;
    //preassign 
    csr.row_ptr = (int *)malloc((l+1) * sizeof(int));
    csr.col_index = (int *)malloc((l+1) * sizeof(int));
    csr.val = (int *)malloc((l+1) * sizeof(int));
    csr.rpl = rpl;

    for(int i=0;i<l;i++){
        csr.col_index[i]=ci[i];
        csr.val[i]=v[i];
    }
    for(int i=0;i<rpl;i++){
        csr.row_ptr[i]=rp[i];
    }
}

void printcsr(const sparse_csr & csr){
    printf("csr row_ptr:\n");
    for(int i=0;i<csr.rpl;i++){
        printf("%d ",csr.row_ptr[i]);
    }
    printf("\n");

    printf("csr col_index:\n");
    for(int i=0;i<csr.len;i++){
        printf("%d ",csr.col_index[i]);
    }
    printf("\n");

    printf("csr val:\n");
    for(int i=0;i<csr.len;i++){
        printf("%d ",csr.val[i]);
    }
    printf("\n");
}

#endif