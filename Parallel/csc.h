#ifndef _CSC_H
# define _CSC_H

typedef struct csc_describe{
    int* col_ptr;
    int* row_index;
    int* val;
    int len;
    int cpl;
} sparse_csc;

void initcsc(sparse_csc & csr, int l, int *cp, int *ri, int *v, int cpl){
    csr.len=l;
    //preassign 
    csr.col_ptr = (int *)malloc((l+1) * sizeof(int));
    csr.row_index = (int *)malloc((l+1) * sizeof(int));
    csr.val = (int *)malloc((l+1) * sizeof(int));
    csr.cpl = cpl;

    for(int i=0;i<l;i++){
        csr.row_index[i]=ri[i];
        csr.val[i]=v[i];
    }
    for(int i=0;i<cpl;i++){
        csr.col_ptr[i]=cp[i];
    }
}

void printcsc(const sparse_csc & csr){
    printf("csr col_ptr:\n");
    for(int i=0;i<csr.cpl;i++){
        printf("%d ",csr.col_ptr[i]);
    }
    printf("\n");

    printf("csr row_index:\n");
    for(int i=0;i<csr.len;i++){
        printf("%d ",csr.row_index[i]);
    }
    printf("\n");

    printf("csr val:\n");
    for(int i=0;i<csr.len;i++){
        printf("%d ",csr.val[i]);
    }
    printf("\n");
}

#endif