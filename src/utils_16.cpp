/*
 * utils_16.cpp — retained as an empty translation unit.
 *
 * All 16-bit ↔ fp32 conversions are `static inline` definitions in
 * utils_16.h, so they inline into every translation unit that uses
 * them (no LTO required).  This file exists only so Makefiles that
 * list it in their sources keep working; it defines nothing.
 */

#include "utils_16.h"
