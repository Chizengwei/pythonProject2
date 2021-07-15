#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
  double x;
  double y;
  double max_x;
  unsigned int mirror;
  double rotation;
  unsigned int valid;
} CPoint;

void add_segments(const double *iarr, uintptr_t length);

void clear_segments(void);

void confirm_placement(void);

CPoint find_segments(const double *iarr,
                     uintptr_t length,
                     uint64_t rotations,
                     bool mirror,
                     double maxx,
                     double maxy);

void info(void);

void init_segments(uint16_t scanlines);
