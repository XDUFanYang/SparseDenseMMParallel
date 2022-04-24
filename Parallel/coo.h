#ifndef _COO_H
# define _COO_H

#include<stdio.h>
// row_index如何动态调整长度？
typedef struct coo_describe{
    int len;
    int* row_index;
    int* col_index;
    int* val;
} sparse_coo;

void insert(int r_index, int c_index, int v, sparse_coo & sc){
    sc.row_index[sc.len] = r_index;
    sc.col_index[sc.len] = c_index;
    sc.val[sc.len] = v;
    sc.len++;
}

void init(sparse_coo & sc, int index_len){
    sc.len=0;
    //preassign 
    sc.row_index = (int *)malloc(index_len* sizeof(int));
    sc.col_index = (int *)malloc(index_len* sizeof(int));
    sc.val = (int *)malloc(index_len* sizeof(int));
}

void printcoo(const sparse_coo & sc){
    printf("coo len:%d \n",sc.len);

    printf("coo row_index:\n");
    for(int i=0;i<sc.len;i++){
        printf("%d ",sc.row_index[i]);
    }
    printf("\n");

    printf("coo col_index:\n");
    for(int i=0;i<sc.len;i++){
        printf("%d ",sc.col_index[i]);
    }
    printf("\n");

    printf("coo val:\n");
    for(int i=0;i<sc.len;i++){
        printf("%d ",sc.val[i]);
    }
    printf("\n");
}

#endif