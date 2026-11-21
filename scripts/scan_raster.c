#include <math.h>
#include <stdint.h>

void raster(int width, int height, int count, const float *screen,
            const uint32_t *faces, const float *attributes, const uint8_t *material,
            float *depth, float *fragments, uint8_t *surfaces, int shade_only) {
    for (int i = 0; i < count; i++) {
        uint32_t a = faces[3*i], b = faces[3*i+1], c = faces[3*i+2];
        const float *p = screen + 3*a, *q = screen + 3*b, *r = screen + 3*c;
        float den = (q[1]-r[1])*(p[0]-r[0])+(r[0]-q[0])*(p[1]-r[1]);
        if (fabsf(den) < 0.00001f) continue;
        int x0 = fmaxf(0, floorf(fminf(p[0], fminf(q[0], r[0]))));
        int x1 = fminf(width-1, ceilf(fmaxf(p[0], fmaxf(q[0], r[0]))));
        int y0 = fmaxf(0, floorf(fminf(p[1], fminf(q[1], r[1]))));
        int y1 = fminf(height-1, ceilf(fmaxf(p[1], fmaxf(q[1], r[1]))));
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                float dx = x + 0.5f - r[0], dy = y + 0.5f - r[1];
                float u = ((q[1]-r[1])*dx+(r[0]-q[0])*dy)/den;
                float v = ((r[1]-p[1])*dx+(p[0]-r[0])*dy)/den;
                float w = 1-u-v;
                if (u < -0.0001f || v < -0.0001f || w < -0.0001f) continue;
                float z = u*p[2]+v*q[2]+w*r[2];
                int j = y*width+x;
                if (z <= depth[j]) continue;
                depth[j] = z;
                if (shade_only) continue;
                const float *aa = attributes+8*a, *bb = attributes+8*b, *cc = attributes+8*c;
                for (int k = 0; k < 8; k++) fragments[8*j+k] = u*aa[k]+v*bb[k]+w*cc[k];
                surfaces[j] = material[i];
            }
        }
    }
}
