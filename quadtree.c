#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// using namespace std;

unsigned long long int green;
unsigned long long int blue;
unsigned long long int red;
unsigned long long int mean;
char magic[3];
int color;

uint32_t min(int a, int b)
{
	return a<b?a:b;
}

typedef struct colornode{

	unsigned char red;
	unsigned char green;
	unsigned char blue;

}node;

typedef struct Quadtree {
	
	node colornode;
	long long index;	
	uint32_t area;

	struct Quadtree *top_left,    *top_right;
	struct Quadtree *bottom_left, *bottom_right;
 
 }Quadtree;

void compress(node ** matrix, Quadtree ** nod, int x, int y, int size, int ratio)
{
	int i, j;
	green=0;red=0;blue=0;mean=0;
	int sizesquare = (size*size);
	unsigned long long int temp1;
	unsigned long long int temp2;
	unsigned long long int temp3;
	(*nod) = malloc(sizeof(Quadtree));
	(*nod)->area = size*size;
	for(i = y; i < y + size; i++){
		for(j = x; j < x + size; j++){

			green = green + matrix[i][j].green;
			red   = red   + matrix[i][j].red;
			blue  = blue  + matrix[i][j].blue;
		}
	}

	red=  red/sizesquare;
	green=green/sizesquare;
	blue= blue/sizesquare;
	(*nod)->colornode.green = green;
	(*nod)->colornode.blue  = blue;
	(*nod)->colornode.red   = red;
	
	for(i = y; i < y + size; i++){
		for(j = x; j < x + size; j++)
		{	
			temp1 =  red - matrix[i][j].red;
			temp2 =  green - matrix[i][j].green;
			temp3 =  blue - matrix[i][j].blue;
			mean = mean + (temp1*temp1) + (temp2*temp2) + (temp3*temp3);
		}
	}
	mean = mean / (3*sizesquare);
	if(mean <= ratio)
	{
		(*nod)->top_right    = NULL;
		(*nod)->top_left     = NULL;
		(*nod)->bottom_left  = NULL;
		(*nod)->bottom_right = NULL;
		return;
	}
	else
	{
		compress(matrix, &(*nod)->top_left,x,y,size/2,ratio);
		compress(matrix, &(*nod)->top_right,x + (size/2), y,size/2, ratio);
		compress(matrix, &(*nod)->bottom_right, x + (size/2), y + (size/2), size/2, ratio);
		compress(matrix, &(*nod)->bottom_left,x,y + (size/2), size/2, ratio);
		return;
	}
}

void decompress(Quadtree * nod, node *** matrix, int x, int y, int size)
{

	if(nod->top_left == NULL && nod->top_right == NULL && nod->bottom_right == NULL && nod->bottom_left == NULL)
	{
		for(int i = y; i < y + size; i++)
			for(int j = x; j < x + size; j++)
			{
				(*matrix)[i][j].red   = nod->colornode.red;
				(*matrix)[i][j].green = nod->colornode.green;
				(*matrix)[i][j].blue  = nod->colornode.blue;
			}
	}
	else
	{
		
		decompress(nod->top_left,matrix, x,y,size/2);
		decompress(nod->top_right,matrix, x+(size/2), y,size/2);
		decompress(nod->bottom_right,matrix, x+(size/2), y+(size/2), size/2);
		decompress(nod->bottom_left,matrix, x,y+(size/2), size/2);
	}

	free(nod);
}

void freeeverything(node **mat,node **r,int width,int height){
	for(int j = 0; j < width; j++) free(mat[j]);
	for(int j = 0; j < height; j++)	free(r[j]);
	free(mat);
	free(r);
}

int main(int argc, char * argv[])
{
	
	int ratio, width, height;
	Quadtree * nod = NULL;
	ratio = atoi(argv[1]);
	FILE * f1;
	f1 = fopen(argv[2], "rb");
	fscanf(f1, "%s"   ,magic);
	fscanf(f1, "%d "  ,&width);
	fscanf(f1, "%d\n" ,&height);
	fscanf(f1, "%d"   ,&color);
	char garbage;
	fread(&garbage,sizeof(char),1,f1);
	
	node ** r = (node**) malloc(sizeof(node*) * (height));
	
	for(int j = 0; j < (height); j++)
	{	
		r[j] = malloc(sizeof(node) * (width));
		fread(r[j], sizeof(node), (width), f1);
	}
	fclose(f1);
	
	compress(r, &nod, 0, 0, width, ratio);				
	
	node ** mat = (node**)malloc(sizeof(node*) * width);
	
	for(unsigned int i = 0; i < width; i++){
		mat[i] = malloc(sizeof(node) * width);
	}

	decompress(nod, &mat, 0, 0, width);

	char *fi = argv[3];
	FILE * f = fopen(fi, "w");
	fprintf(f, "P6\n");
	fprintf(f, "%d %d\n",width,width);
	fprintf(f, "255\n");


	for(int j = 0;j<width; j++) fwrite(mat[j], sizeof(node),width,f);
	fclose(f);

	freeeverything(mat,r,width,height);
	
}
