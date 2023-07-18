#include <math.h>
#include "image.h"

float nn_interpolate(image im, float x, float y, int c)
{
  return get_pixel(im, round(x), round(y), c);
}

image nn_resize(image im, int w, int h)
{
  image ret = make_image(w, h, im.c);
  float a_x = (float)im.w / w;
  float a_y = (float)im.h / h;
  float b_x = -0.5 + 0.5*a_x;
  float b_y = -0.5 + 0.5*a_y;
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) {
      for (int c = 0; c < im.c; c++) {
        float x_new = a_x*x + b_x;
        float y_new = a_y*y + b_y;
        ret.data[x + y*w + c*w*h] = nn_interpolate(im, x_new, y_new, c);
      }
    }
  }
  return ret;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
  int left = floorf(x);
  int right = ceilf(x);
  int top = floorf(y);
  int bottom = ceilf(y);
  float v1 = get_pixel(im, left, top, c);
  float v2 = get_pixel(im, right, top, c);
  float v3 = get_pixel(im, left, bottom, c);
  float v4 = get_pixel(im, right, bottom, c);
  float d1 = bottom - y;
  float d2 = y - top;
  float d3 = x - left;
  float d4 = right - x;
  float q1 = v1*d1 + v3*d2;
  float q2 = v2*d1 + v4*d2;
  float q = q2*d3 + q1*d4;
  return q;
}

image bilinear_resize(image im, int w, int h)
{
  image ret = make_image(w, h, im.c);
  float a_x = (float)im.w / w;
  float a_y = (float)im.h / h;
  float b_x = -0.5 + 0.5*a_x;
  float b_y = -0.5 + 0.5*a_y;
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) {
      for (int c = 0; c < im.c; c++) {
        float x_new = a_x*x + b_x;
        float y_new = a_y*y + b_y;
        ret.data[x + y*w + c*w*h] = bilinear_interpolate(im, x_new, y_new, c);
      }
    }
  }
  return ret;
}
