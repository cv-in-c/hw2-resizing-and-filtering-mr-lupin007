#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"

#define M_PI 3.14159265358979323846


void l1_normalize(image im)
{
  float sum = 0;
  for (int x = 0; x < im.w; x++) {
    for (int y = 0; y < im.h; y++) {
      for (int c = 0; c < im.c; c++) {
        sum += im.data[x + y*im.w + c*im.w*im.h];
      }
    }
  }
  for (int x = 0; x < im.w; x++) {
    for (int y = 0; y < im.h; y++) {
      for (int c = 0; c < im.c; c++) {
        float value;
        if (sum != 0) {
          value = im.data[x + y*im.w + c*im.w*im.h] / sum;
        }
        else {
          value = 1.0 / (im.w*im.h);
        }
        im.data[x + y*im.w + c*im.w*im.h] = value;
      }
    }
  }
}

image make_box_filter(int w)
{
  image ret = make_image(w, w, 1);
  for (int i = 0; i < w*w; i++) {
    ret.data[i] = 1.0;
  }
  l1_normalize(ret);
  return ret;
}

image convolve_image(image im, image filter, int preserve)
{
  assert(filter.c == im.c || filter.c == 1);
  image ret = make_image(im.w, im.h, im.c);
  for (int x = 0; x < im.w; x++) {
    for (int y = 0; y < im.h; y++) {
      for (int c = 0; c < im.c; c++) {
        float q = 0;
        int fil;
        if (filter.c != 1) {
          fil = c;
        }
        else {
          fil = 0;
        }
        for (int filter_x = 0; filter_x < filter.w; filter_x++) {
          for (int filter_y = 0; filter_y < filter.h; filter_y++) {
            float value = filter.data[filter_x + filter_y*filter.w + filter.w*filter.h*fil];
            int fx = x - filter.w / 2 + filter_x;
            int fy = y - filter.h / 2 + filter_y;
            q += get_pixel(im, fx, fy, c) * value;
          }
        }
        ret.data[x + y*im.w + c*im.w*im.h] = q;
      }
    }
  }
  if (!preserve) {
    image ret_ = make_image(im.w, im.h, 1);
    for (int w = 0; w < im.w; w++) {
      for (int h = 0; h < im.h; h++) {
        float q = 0;
        for (int c = 0; c < im.c; c++) {
          q += ret.data[w + h*im.w + im.w*im.h*c];
        }
        ret_.data[w + h*im.w] = q;
      }
    }
    return ret_;
  }
  else {
    return ret;
  }
}

image make_highpass_filter()
{
  image ret = make_box_filter(3);
  ret.data[0] = 0;
  ret.data[1] = -1;
  ret.data[2] = 0;
  ret.data[3] = -1;
  ret.data[4] = 4;
  ret.data[5] = -1;
  ret.data[6] = 0;
  ret.data[7] = -1;
  ret.data[8] = 0;
  return ret;
}

image make_sharpen_filter()
{
  image ret = make_box_filter(3);
  ret.data[0] = 0;
  ret.data[1] = -1;
  ret.data[2] = 0;
  ret.data[3] = -1;
  ret.data[4] = 5;
  ret.data[5] = -1;
  ret.data[6] = 0;
  ret.data[7] = -1;
  ret.data[8] = 0;
  return ret;
}

image make_emboss_filter()
{
  image ret = make_box_filter(3);
  ret.data[0] = -2;
  ret.data[1] = -1;
  ret.data[2] = 0;
  ret.data[3] = -1;
  ret.data[4] = 1;
  ret.data[5] = 1;
  ret.data[6] = 0;
  ret.data[7] = 1;
  ret.data[8] = 2;
  return ret;
}

// Question 2.2.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: We should use preserve for sharpen and emboss because we want to preserve their colors.
// Question 2.2.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: Post-processing, especially clamping, was needed for all of the above filters so that the colors will not be out of bounds.

image make_gaussian_filter(float sigma)
{
  int s = (int)roundf(sigma * 6) + 1;
  int w;
  if (s % 2 != 0) {
    w = s;
  }
  else {
    w = s++;
  }
  image ret = make_box_filter(w);
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < w; y++) {
      int center = w / 2;
      float value = 1 / (sigma*sigma*2*M_PI);
      float power = -((x-center)*(x-center)+(y-center)*(y-center)) / (sigma*sigma*2);
      ret.data[x + y*w] = value * exp(power);
    }
  }
  l1_normalize(ret);
  return ret;
}

image add_image(image a, image b)
{
  assert(a.w == b.w && a.h == b.h && a.c == b.c);
  image ret = make_image(a.w, a.h, a.c);
  for (int x = 0; x < a.w; x++) {
    for (int y = 0; y < a.h; y++) {
      for (int c = 0; c < a.c; c++) {
        int pixel = x + y*a.w + c*a.w*a.h;
        ret.data[pixel] = a.data[pixel] + b.data[pixel];
      }
    }
  }
  return ret;
}

image sub_image(image a, image b)
{
  assert(a.w == b.w && a.h == b.h && a.c == b.c);
  image ret = make_image(a.w, a.h, a.c);
  for (int x = 0; x < a.w; x++) {
    for (int y = 0; y < a.h; y++) {
      for (int c = 0; c < a.c; c++) {
        int pixel = x + y*a.w + c*a.w*a.h;
        ret.data[pixel] = a.data[pixel] - b.data[pixel];
      }
    }
  }
  return ret;
}

image make_gx_filter()
{
  image ret = make_box_filter(3);
  ret.data[0] = -1;
  ret.data[1] = 0;
  ret.data[2] = 1;
  ret.data[3] = -2;
  ret.data[4] = 0;
  ret.data[5] = 2;
  ret.data[6] = -1;
  ret.data[7] = 0;
  ret.data[8] = 1;
  return ret;
}

image make_gy_filter()
{
  image ret = make_box_filter(3);
  ret.data[0] = -1;
  ret.data[1] = -2;
  ret.data[2] = -1;
  ret.data[3] = 0;
  ret.data[4] = 0;
  ret.data[5] = 0;
  ret.data[6] = 1;
  ret.data[7] = 2;
  ret.data[8] = 1;
  return ret;
}

void feature_normalize(image im)
{
  float max = -1.0;
  float min = INFINITY;
  for (int x = 0; x < im.w; x++) {
    for (int y = 0; y < im.h; y++) {
      for (int c = 0; c < im.c; c++) {
        int i = x + y*im.w + c*im.w*im.h;
        if (im.data[i] > max) {
          max = im.data[i];
        }
        if (im.data[i] < min) {
          min = im.data[i];
        }
      }
    }
  }
  if (max - min != 0) {
    for (int x = 0; x < im.w; x++) {
      for (int y = 0; y < im.h; y++) {
        for (int c = 0; c < im.c; c++) {
          int i = x + y*im.w + c*im.w*im.h;
          im.data[i] = (im.data[i] - min) / (max - min);
        }
      }
    }
  }
  else {
    for (int x = 0; x < im.w; x++) {
      for (int y = 0; y < im.h; y++) {
        for (int c = 0; c < im.c; c++) {
          int i = x + y*im.w + c*im.w*im.h;
          im.data[i] = 0;
        }
      }
    }
  }
}

image *sobel_image(image im)
{
  image filter_gx = make_gx_filter();
  image filter_gy = make_gy_filter();
  image gx = convolve_image(im, filter_gx, 0);
  image gy = convolve_image(im, filter_gy, 0);
  image *ret = calloc(2, sizeof(image));
  ret[0] = make_image(im.w, im.h, 1);
  ret[1] = make_image(im.w, im.h, 1);
  for (int x = 0; x < im.w; x++) {
    for (int y = 0; y < im.h; y++) {
      int i = x + y*im.w;
      float magnitude = sqrtf(gx.data[i]*gx.data[i] + gy.data[i]*gy.data[i]);
      float gradient = atan2(gy.data[i], gx.data[i]);
      ret[0].data[i] = magnitude;
      ret[1].data[i] = gradient;
    }
  }
  return ret;
}

image colorize_sobel(image im)
{
  image *ret = sobel_image(im);
  feature_normalize(ret[0]);
  feature_normalize(ret[1]);
  image color = make_image(im.w, im.h, 3);
  for (int x = 0; x < color.w; x++) {
    for (int y = 0; y < color.h; y++) {
      for (int c = 0; c < color.c; c++) {
        if (c) {
          color.data[x + y*im.w + c*im.w*im.h] = ret[0].data[x + y*im.w];
        }
        else {
          color.data[x + y*im.w + c*im.w*im.h] = ret[1].data[x + y*im.w];
        }
      }
    }
  }
  hsv_to_rgb(color);
  return convolve_image(color, make_gaussian_filter(1), 1);
}

image bilateral_filter(image im)
{
  image ret = make_image(im.w, im.h, im.c);
  for (int x = 0; x < im.w; x++){
    for (int y = 0; y < im.h; y++){
      for (int c = 0; c < im.c; c++){
      int diameter = 3;
      int half = diameter / 2;
      int sigmad = 16;
      float sigmar = 0.2;
      double sum = 0;
      double value = 0;
      for (int i = 0; i < diameter; i++) {
        for (int j = 0; j < diameter; j++) {
          int neighbor_x = x - (half - i);
          int neighbor_y = y - (half - j);
          double gi = exp(-((im.data[neighbor_x + neighbor_y*im.w + c*im.w*im.h]-im.data[x + y*im.w + c*im.w*im.h])*(im.data[neighbor_x + neighbor_y*im.w + c*im.w*im.h]-im.data[x + y*im.w + c*im.w*im.h]))/(sigmar*sigmar*2))/(sigmar*sigmar*2*M_PI);
          double gs = exp(-((half-i)*(half-i)+(half-j)*(half-j))/(sigmad*sigmad*2))/(sigmad*sigmad*2*M_PI);
          sum += gi*gs;
        }
      }
      for (int i = 0; i < diameter; i++) {
        for (int j = 0; j < diameter; j++) {
          int neighbor_x = x - (half - i);
          int neighbor_y = y - (half - j);
          double gi = exp(-((im.data[neighbor_x + neighbor_y*im.w + c*im.w*im.h]-im.data[x + y*im.w + c*im.w*im.h])*(im.data[neighbor_x + neighbor_y*im.w + c*im.w*im.h]-im.data[x + y*im.w + c*im.w*im.h]))/(sigmar*sigmar*2))/(sigmar*sigmar*2*M_PI);
          double gs = exp(-((half-i)*(half-i)+(half-j)*(half-j))/(sigmad*sigmad*2))/(sigmad*sigmad*2*M_PI);
          value += im.data[neighbor_x + neighbor_y*im.w + c*im.w*im.h]*((gi*gs)/sum);
        }
      }
      ret.data[x + y*im.w + c*im.w*im.h] = value;
      }
    }
  }
  return ret;
}
