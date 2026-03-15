
#include "ringchecker_step.h"


int max_perimeter = 32000;
int max_radius = 1024;
int max_manhatten =
    2048; // perimeter containing point is unlikely to stray too far from island

const int large = 0;
const int biome_scale = 4;

#ifndef MOCK_MOOSHROOM
int is_mooshroom(BiomeNoise *bn, int x, int z, double threshhold) {
  double cont = sampleClimatePara(bn, NULL, (double)x / biome_scale,
                                  (double)z / biome_scale);
  // printf("is_m  x,z,c : %d, %d -> %f\n", x, z, cont);
  // printf("double conversion: %f, %f\n", (double) x, (double) z);
  return cont < threshhold;
}
#endif

int manhatten_dist(int x1, int z1, int x2, int z2) {
  return abs(x2 - x1) + abs(z2 - z1);
}

// converts line[i] index to vector step
int ltovecz(int q) {
  if (q == 0) {
    return 1;
  }
  if (q == 2) {
    return -1;
  }
  return 0;
}
int ltovecx(int q) {
  if (q == 1) {
    return -1;
  }
  if (q == 3) {
    return 1;
  }
  return 0;
}
int vectol(int x, int z) {
  if (x == 1) {
    return 3;
  }
  if (x == -1) {
    return 1;
  }
  if (z == 1) {
    return 0;
  }
  if (z == -1) {
    return 2;
  }
  printf("ILLEGAL (x,z)!");
  return 0;
}

int intersects_ray(int px, int pz, int sx, int sz, int udx, int udz, int step) {
  // check if perimeter crosses pos-x ray ( offset 0.5 blocks in z-axis )
  return (px > sx                    // ray
          && udx == 0                // perimeter perpendicular to ray
          && ((pz == sz && udz == 1) // crossing from below
              || (pz == sz + step && udz == -1))); // crossing from above
}
void populate_quadrants(int *q, BiomeNoise *bn, int px, int pz, int udx,
                        int udz, int step, double threshhold) {
  // to avoid repeat samples, we shift previous quadrants by (-dx, -dz)
  // use dx == dz == 0 to initialise all 4 quadrants
  if (udx == 0 && udz == 0) {
    q[0] = is_mooshroom(bn, px + udx * step, pz + udz * step, threshhold);
    q[1] = is_mooshroom(bn, px + (udx - 1) * step, pz + udz * step, threshhold);
    q[2] = is_mooshroom(bn, px + (udx - 1) * step, pz + udz - 1, threshhold);
    q[3] = is_mooshroom(bn, px + udx * step, pz + (udz - 1) * step, threshhold);
  } else {
    if (udx == 1) {
      q[1] = q[0];
      q[2] = q[3];
      q[0] = is_mooshroom(bn, px + udx * step, pz + udz * step, threshhold);
      q[3] =
          is_mooshroom(bn, px + udx * step, pz + (udz - 1) * step, threshhold);
    } else if (udx == -1) {
      q[0] = q[1];
      q[3] = q[2];
      q[1] =
          is_mooshroom(bn, px + (udx - 1) * step, pz + udz * step, threshhold);
      q[2] = is_mooshroom(bn, px + (udx - 1) * step, pz + (udz - 1) * step,
                          threshhold);
    } else if (udz == 1) {
      q[2] = q[1];
      q[3] = q[0];
      q[0] = is_mooshroom(bn, px + udx * step, pz + udz * step, threshhold);
      q[1] =
          is_mooshroom(bn, px + (udx - 1) * step, pz + udz * step, threshhold);
    } else if (udz == -1) {
      q[1] = q[2];
      q[0] = q[3];
      q[2] = is_mooshroom(bn, px + (udx - 1) * step, pz + (udz - 1) * step,
                          threshhold);
      q[3] =
          is_mooshroom(bn, px + udx * step, pz + (udz - 1) * step, threshhold);
    }
  }
}
void find_valid_paths(int *l, int *q) {
  for (int i = 0; i < 4; i++) {
    // == 1 when adj blocks are different (ie. line is on perimeter)
    l[i] = q[i] ^ q[(i + 1) % 4];
  }
}

void take_step(int *px, int *pz, int *udx, int *udz, int path_index, int step) {
  *px += *udx * step;
  *pz += *udz * step;
  *udx = ltovecx(path_index);
  *udz = ltovecz(path_index);
}

int walk_perimeter(BiomeNoise *bn, int *out, int pstartx, int pstartz, int sx,
                   int sz, int step, double threshhold) {
  // out[0] is number of intersections
  // out[1] is max x coord of perimeter (to continue expansion from)
  // out[2] is corresponding z coord
  out[0] = 0;
  out[1] = pstartx;
  out[2] = pstartz;
  int px = pstartx;
  int pz = pstartz;
  int udx = 0, udz = 1; // unit dx and dz - does not consider step size
  int q[4]; // 4 blocks around block corner (anticlockwise from corner of
            // perimeter block)
  int l[4]; // 4 lines from corner (l[i] is line between q[i] and q[(i+1)%4]
  int l_tot;
  int p_len = 0;
  populate_quadrants(q, bn, px, pz, 0, 0, step, threshhold);
  while (p_len++ <= max_perimeter &&
         manhatten_dist(px, pz, sx, sz) <= max_manhatten) {

    out[0] += intersects_ray(px, pz, sx, sz, udx, udz, step);
    populate_quadrants(q, bn, px, pz, udx, udz, step, threshhold);
    find_valid_paths(l, q);
    l_tot = l[0] + l[1] + l[2] + l[3];

    if (l_tot == 4) {
      // DIAGONAL PATTERN
      // find i of vectol[+dx, +dz]
      // find direction of mooshroom block adj
      // choose that direction
      int heading = vectol(udx, udz);
      int turn;
      int check = heading - 1;
      if (check < 0) {
        check = 4 + check;
      }
      if (q[check]) {
        turn = check;
      } else {
        turn = (check + 2) % 4;
      }
      take_step(&px, &pz, &udx, &udz, turn, step);
    } else {
      // choose l[i] that has i different to vectol(-dx,-dz)
      // (no backtracks)
      int backtrack = vectol(-udx, -udz);
      int p_found = 0;
      for (int i = 0; i < 4; i++) {
        if (i != backtrack && l[i]) {
          take_step(&px, &pz, &udx, &udz, i, step);
          p_found = 1;
          break;
        }
      }
      if (!p_found) {
        printf("path not found! point is likely inside mooshroom island ( %d, "
               "%d ) \n",
               px, pz);
        return 1;
      }
    }

    // update point to continue from if perimeter does not surround island
    if (px > out[1]) {
      out[1] = px;
      out[2] = pz;
    }

    if (px == pstartx && pz == pstartz // back to starting point
        &&
        udz == 1) { // avoid case where starting point is diagonally connected
      return 0;
    }
  }

  // printf("perimeter too large if lenght: %d >= %d\n", p_len, max_perimeter);
  // printf("perimeter strays too far if distance: %d >= %d\n",
  //   manhatten_dist(px,pz,sx,sz), max_manhatten);
  return 1;
}

int is_surrounded(BiomeNoise *bn, int startx, int startz, int step,
                  double threshhold) {
  int dx = 0;
  int dz = 0;
  int p_res[3];
  if (is_mooshroom(bn, startx, startz,
                   threshhold)) { // to be surrounded, must not BE mooshroom!!
    return 0;
  }
  while ((dx += step) < max_radius) {
    if (is_mooshroom(bn, startx + dx, startz + dz, threshhold)) {
      // loop circumference
      int err = walk_perimeter(bn, p_res, startx + dx, startz + dz, startx,
                               startz, step, threshhold);
      if (err) {
        // printf("Error checking surrounded, s = (%d, %d); dx = %d\n", startx,
        //       startz, dx);
        return 0;
      }
      if (p_res[0] % 2 == 0) {
        // point not inside perimeter
        // restart scan from point furthest +x
        dx = p_res[1] - startx - step;
        dz = p_res[2] - startz - step;
      } else {
        return 1; // is_surrounded == true
      }
    }
  }
  return 0;
}
