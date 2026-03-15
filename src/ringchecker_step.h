#pragma once

#include "cubiomes.h"
#include "../cubiomes/biomenoise.h"
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

extern int max_perimeter;
extern int max_radius ;
extern int max_manhatten ;
extern const int large ;
extern const int biome_scale ;

int is_mooshroom(BiomeNoise *bn, int x, int z, double threshhold);
int walk_perimeter(BiomeNoise *bn, int *out, int pstartx, int pstartz, int sx,
                   int sz, int step, double threshhold);
int is_surrounded(BiomeNoise *bn, int startx, int startz, int step, double threshhold);

#ifdef __cplusplus
}
#endif
