/*
 * libvips C benchmark runner.
 *
 * Executes a pipeline scenario (thumbnail, resize, etc.) on a real image file
 * and reports hard metrics as JSON to stdout.
 *
 * Usage: ./libvips-runner <input_path> <operation> [op_args...] --iterations N --threads N
 *
 * Metrics reported:
 *   - wall_ns[]        per-iteration wall-clock nanoseconds
 *   - alloc_count      total malloc calls (via interposition, if enabled)
 *   - alloc_bytes      total bytes allocated
 *   - peak_rss_kb      max resident set size
 *   - minor_faults     minor page faults
 *   - major_faults     major page faults
 *   - vol_ctx_sw       voluntary context switches
 *   - invol_ctx_sw     involuntary context switches
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <sys/resource.h>
#include <vips/vips.h>

#define MAX_ITERATIONS 10000
#define DEFAULT_ITERATIONS 50
#define DEFAULT_AFFINE_A 1.0
#define DEFAULT_AFFINE_B 0.2
#define DEFAULT_AFFINE_C -0.1
#define DEFAULT_AFFINE_D 0.95
#define DEFAULT_SIMILARITY_SCALE 0.9
#define DEFAULT_SIMILARITY_ANGLE 15.0
#define DEFAULT_GAMMA_EXPONENT 2.4
#define DEFAULT_MAPIM_DX 0.25
#define DEFAULT_MAPIM_DY 0.25
#define MAX_COLOURSPACE_STEPS 8
#define EMBED_OFFSET_X 64
#define EMBED_OFFSET_Y 48
#define EMBED_PAD_WIDTH 256
#define EMBED_PAD_HEIGHT 192
#define EXTRACT_OFFSET_X 32
#define EXTRACT_OFFSET_Y 24
#define EXTRACT_TRIM_WIDTH (EXTRACT_OFFSET_X * 2)
#define EXTRACT_TRIM_HEIGHT (EXTRACT_OFFSET_Y * 2)

static long long timespec_diff_ns(struct timespec *start, struct timespec *end) {
    return (long long)(end->tv_sec - start->tv_sec) * 1000000000LL
         + (long long)(end->tv_nsec - start->tv_nsec);
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <input> <op> [op_args...] --iterations N --threads N [--libvips-cache] [--e2e] [--quiet]\n", prog);
    fprintf(stderr, "Operations: load, load-jpeg (alias: load_jpeg), load-tiff (alias: load_tiff), save-avif, save-exr, save-gif, save-heif, save-jpeg, save-jp2k, save-tiff [none|lzw|deflate|packbits|jpeg], thumbnail <width>, resize <scale>, zoom <xfac> [yfac], shrink <hfactor> [vfactor], shrinkh <factor>, shrinkv <factor>, invert, abs, sign, round, floor, ceil, invert_invert, bandmean, add, multiply, subtract, and, equal, linear <scale> <offset>, cast [u8|u16|f32], flip [horizontal|vertical], gamma [exponent], gauss_blur <sigma>, convolve, sobel, prewitt, laplacian, median_blur <size>, unsharp_mask <sigma> <strength>, colourspace [dest...], srgb_to_lab, affine [a b c d], similarity [scale angle], mapim [dx dy], composite [over|atop], dilate <size>, erode <size>, open <size>, close <size>, sharpen <sigma> <strength>, histogram, recomb, grey, draw_line, freqfilt, conv_sharpen3, conv_sobel3, workflow <target_format> [width], thumbnail_sharpen, thumbnail_colourspace_cast, thumbnail_gauss_blur, thumbnail_linear, resize_colourspace, embed, extract-area (alias: extract_area), embed_extract, three_op_chain, perceptual_enhance [target_format] [width]\n");
    fprintf(stderr, "  --threads N   Pin libvips worker concurrency to N\n");
    fprintf(stderr, "  --e2e   Include full decode-from-disk in every iteration (productive pipeline)\n");
    fprintf(stderr, "  --quiet Suppress JSON output after the run\n");
}

static const char *canonicalize_op_name(const char *op_name) {
    if (strcmp(op_name, "extract_area") == 0)
        return "extract-area";
    if (strcmp(op_name, "load_jpeg") == 0)
        return "load-jpeg";
    if (strcmp(op_name, "load_tiff") == 0 || strcmp(op_name, "load-tiff") == 0)
        return "load";

    return op_name;
}

struct input_blob {
    void *raw_buf;
    size_t raw_len;
    int width;
    int height;
    int bands;
    VipsBandFormat format;
    VipsInterpretation interpretation;
};

static VipsImage *new_input_image(const struct input_blob *input_blob) {
    VipsImage *image = vips_image_new_from_memory(input_blob->raw_buf,
                                                  input_blob->raw_len,
                                                  input_blob->width,
                                                  input_blob->height,
                                                  input_blob->bands,
                                                  input_blob->format);
    if (image)
        image->Type = input_blob->interpretation;
    return image;
}

/* Discard callback for vips_sink_disc: accepts every region and returns success. */
static int discard_tile_callback(VipsRegion *region, VipsRect *area, void *a) {
    (void)region;
    (void)area;
    (void)a;
    return 0;
}

/*
 * Force full evaluation of `out` using a discard sink (no output buffer).
 * This is the correct no-E2E evaluation sink: it forces all tiles to be
 * computed (fulfilling demand) without allocating a copy of the full image.
 *
 * Returns 0 on success, -1 on failure.
 */
static int vips_sink_discard(VipsImage *out) {
    return vips_sink_disc(out, discard_tile_callback, NULL);
}

static int run_thumbnail(const struct input_blob *input_blob, int width) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_thumbnail_image(in, &out, width, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    /* Force pixel computation without allocating a full output buffer. */
    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_composite_e2e(const char *input, VipsBlendMode mode) {
    VipsImage *base = vips_image_new_from_file(input, NULL);
    if (!base) return -1;

    VipsImage *out = NULL;
    int ret = vips_composite2(base, base, &out, mode, NULL);
    g_object_unref(base);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_resize(const struct input_blob *input_blob, double scale) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_resize(in, &out, scale, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_zoom(const struct input_blob *input_blob, int xfac, int yfac) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_zoom(in, &out, xfac, yfac, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static VipsBlendMode parse_composite_mode_arg(const char *arg) {
    if (!arg || strcmp(arg, "over") == 0)
        return VIPS_BLEND_MODE_OVER;
    if (strcmp(arg, "atop") == 0)
        return VIPS_BLEND_MODE_ATOP;

    fprintf(stderr, "composite only accepts optional mode arg 'over' or 'atop', got '%s'\n", arg);
    exit(1);
}

static int run_composite(const struct input_blob *input_blob, VipsBlendMode mode) {
    VipsImage *base = new_input_image(input_blob);
    if (!base) return -1;

    VipsImage *out = NULL;
    int ret = vips_composite2(base, base, &out, mode, NULL);
    g_object_unref(base);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_affine(const struct input_blob *input_blob,
                      double a, double b, double c, double d) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    VipsArea *oarea = VIPS_AREA(vips_array_int_newv(
        4, 0, 0, input_blob->width, input_blob->height));
    int ret = vips_affine(in, &out, a, b, c, d, "oarea", oarea, NULL);
    vips_area_unref(oarea);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_similarity(const struct input_blob *input_blob,
                          double scale, double angle) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_similarity(in, &out, "scale", scale, "angle", angle, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_mapim(const struct input_blob *input_blob, const float *index_buf) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *index = vips_image_new_from_memory(index_buf,
                                                  (size_t) input_blob->width * input_blob->height * 2 * sizeof(float),
                                                  input_blob->width,
                                                  input_blob->height,
                                                  2,
                                                  VIPS_FORMAT_FLOAT);
    if (!index) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *out = NULL;
    int ret = vips_mapim(in, &out, index, "extend", VIPS_EXTEND_COPY, NULL);
    g_object_unref(index);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_shrinkh(const struct input_blob *input_blob, int factor) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_shrinkh(in, &out, factor, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_shrinkv(const struct input_blob *input_blob, int factor) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_shrinkv(in, &out, factor, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_shrink(const struct input_blob *input_blob, int hfactor, int vfactor) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_shrink(in, &out, hfactor, vfactor, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_linear(const struct input_blob *input_blob, double scale, double offset) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_linear1(in, &out, scale, offset, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_add(const struct input_blob *input_blob) {
    return run_linear(input_blob, 1.0, 5.0);
}

static int run_multiply(const struct input_blob *input_blob) {
    return run_linear(input_blob, 2.0, 0.0);
}

static int run_subtract(const struct input_blob *input_blob) {
    return run_linear(input_blob, 1.0, -5.0);
}

static double equal_rhs_for_blob(const struct input_blob *input_blob) {
    switch (input_blob->format) {
    case VIPS_FORMAT_USHORT:
        return 32768.0;
    case VIPS_FORMAT_FLOAT:
    case VIPS_FORMAT_DOUBLE:
        return 0.5;
    default:
        return 128.0;
    }
}

static int run_and(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_andimage_const1(in, &out, 240.0, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_equal(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_equal_const1(in, &out, equal_rhs_for_blob(input_blob), NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_histogram(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_hist_find(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_recomb(const struct input_blob *input_blob) {
    if (input_blob->bands != 3) {
        fprintf(stderr, "recomb benchmark expects a 3-band input image, got %d\n", input_blob->bands);
        return -1;
    }

    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    static const double matrix_values[6] = {
        0.299, 0.587, 0.114,
        1.000, 0.000, -1.000
    };
    VipsImage *matrix = vips_image_new_matrix_from_array(3, 2, matrix_values, 6);
    if (!matrix) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *out = NULL;
    int ret = vips_recomb(in, &out, matrix, NULL);
    g_object_unref(matrix);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_grey(int width, int height) {
    VipsImage *out = NULL;
    int ret = vips_grey(&out, width, height, NULL);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_draw_line(int width, int height, int bands) {
    VipsImage *image = NULL;
    int ret = vips_black(&image, width, height, "bands", bands, NULL);
    if (ret != 0 || !image) return -1;

    double ink[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (int i = 0; i < bands && i < 4; i++)
        ink[i] = 1.0;

    ret = vips_draw_line(image,
                         ink,
                         bands,
                         0,
                         height / 2,
                         width > 0 ? width - 1 : 0,
                         height / 2,
                         NULL);
    if (ret != 0) {
        g_object_unref(image);
        return -1;
    }

    int sink_ret = vips_sink_discard(image);
    g_object_unref(image);
    return sink_ret;
}

static int run_draw_rect(int width, int height, int bands) {
    VipsImage *image = NULL;
    int ret = vips_black(&image, width, height, "bands", bands, NULL);
    if (ret != 0 || !image) return -1;

    double ink[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (int i = 0; i < bands && i < 4; i++)
        ink[i] = 1.0;

    const int rect_width = VIPS_MAX(width / 2, 1);
    const int rect_height = VIPS_MAX(height / 2, 1);
    const int left = (width - rect_width) / 2;
    const int top = (height - rect_height) / 2;

    ret = vips_draw_rect(image,
                         ink,
                         bands,
                         left,
                         top,
                         rect_width,
                         rect_height,
                         "fill",
                         TRUE,
                         NULL);
    if (ret != 0) {
        g_object_unref(image);
        return -1;
    }

    int sink_ret = vips_sink_discard(image);
    g_object_unref(image);
    return sink_ret;
}

static int run_draw_circle(int width, int height, int bands) {
    VipsImage *image = NULL;
    int ret = vips_black(&image, width, height, "bands", bands, NULL);
    if (ret != 0 || !image) return -1;

    double ink[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (int i = 0; i < bands && i < 4; i++)
        ink[i] = 1.0;

    ret = vips_draw_circle(image,
                           ink,
                           bands,
                           width / 2,
                           height / 2,
                           VIPS_MAX(VIPS_MIN(width, height) / 4, 1),
                           "fill",
                           FALSE,
                           NULL);
    if (ret != 0) {
        g_object_unref(image);
        return -1;
    }

    int sink_ret = vips_sink_discard(image);
    g_object_unref(image);
    return sink_ret;
}

static int run_freqfilt(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *mono = NULL;
    int ret = input_blob->bands > 1
        ? vips_bandmean(in, &mono, NULL)
        : vips_copy(in, &mono, NULL);
    g_object_unref(in);
    if (ret != 0 || !mono) return -1;

    VipsImage *fft = NULL;
    ret = vips_fwfft(mono, &fft, NULL);
    g_object_unref(mono);
    if (ret != 0 || !fft) return -1;

    VipsImage *out = NULL;
    ret = vips_invfft(fft, &out, "real", TRUE, NULL);
    g_object_unref(fft);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_gauss_blur(const struct input_blob *input_blob, double sigma) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_gaussblur(in, &out, sigma, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_colourspace_chain_image(VipsImage *in,
                                       const VipsInterpretation *targets,
                                       int target_count) {
    VipsImage *current = in;

    for (int i = 0; i < target_count; i++) {
        VipsImage *next = NULL;
        int ret = vips_colourspace(current, &next, targets[i], NULL);
        g_object_unref(current);
        if (ret != 0 || !next)
            return -1;
        current = next;
    }

    void *buf = vips_image_write_to_memory(current, NULL);
    g_object_unref(current);
    g_free(buf);
    return buf ? 0 : -1;
}

static int default_colourspace_route(VipsInterpretation interpretation,
                                     VipsInterpretation *targets) {
    switch (interpretation) {
    case VIPS_INTERPRETATION_B_W:
    case VIPS_INTERPRETATION_GREY16:
        targets[0] = VIPS_INTERPRETATION_sRGB;
        targets[1] = VIPS_INTERPRETATION_B_W;
        return 2;
    case VIPS_INTERPRETATION_scRGB:
        targets[0] = VIPS_INTERPRETATION_XYZ;
        targets[1] = VIPS_INTERPRETATION_scRGB;
        return 2;
    case VIPS_INTERPRETATION_LAB:
        targets[0] = VIPS_INTERPRETATION_sRGB;
        targets[1] = VIPS_INTERPRETATION_LAB;
        return 2;
    case VIPS_INTERPRETATION_XYZ:
        targets[0] = VIPS_INTERPRETATION_LAB;
        targets[1] = VIPS_INTERPRETATION_XYZ;
        return 2;
    case VIPS_INTERPRETATION_CMYK:
        targets[0] = VIPS_INTERPRETATION_sRGB;
        targets[1] = VIPS_INTERPRETATION_CMYK;
        return 2;
    case VIPS_INTERPRETATION_HSV:
        targets[0] = VIPS_INTERPRETATION_sRGB;
        targets[1] = VIPS_INTERPRETATION_HSV;
        return 2;
    case VIPS_INTERPRETATION_LCH:
        targets[0] = VIPS_INTERPRETATION_CMC;
        targets[1] = VIPS_INTERPRETATION_LCH;
        return 2;
    case VIPS_INTERPRETATION_CMC:
        targets[0] = VIPS_INTERPRETATION_LCH;
        targets[1] = VIPS_INTERPRETATION_CMC;
        return 2;
    default:
        targets[0] = VIPS_INTERPRETATION_LAB;
        targets[1] = VIPS_INTERPRETATION_sRGB;
        return 2;
    }
}

static VipsInterpretation parse_colourspace_arg(const char *arg) {
    if (strcasecmp(arg, "srgb") == 0 || strcasecmp(arg, "rgb") == 0)
        return VIPS_INTERPRETATION_sRGB;
    if (strcasecmp(arg, "lab") == 0)
        return VIPS_INTERPRETATION_LAB;
    if (strcasecmp(arg, "xyz") == 0)
        return VIPS_INTERPRETATION_XYZ;
    if (strcasecmp(arg, "yxy") == 0)
        return VIPS_INTERPRETATION_YXY;
    if (strcasecmp(arg, "hsv") == 0)
        return VIPS_INTERPRETATION_HSV;
    if (strcasecmp(arg, "cmyk") == 0)
        return VIPS_INTERPRETATION_CMYK;
    if (strcasecmp(arg, "scrgb") == 0)
        return VIPS_INTERPRETATION_scRGB;
    if (strcasecmp(arg, "greyscale") == 0 || strcasecmp(arg, "grayscale") == 0
        || strcasecmp(arg, "grey") == 0 || strcasecmp(arg, "gray") == 0
        || strcasecmp(arg, "bw") == 0 || strcasecmp(arg, "b-w") == 0)
        return VIPS_INTERPRETATION_B_W;
    if (strcasecmp(arg, "lch") == 0)
        return VIPS_INTERPRETATION_LCH;
    if (strcasecmp(arg, "ucs") == 0 || strcasecmp(arg, "cmc") == 0)
        return VIPS_INTERPRETATION_CMC;

    fprintf(stderr,
            "colourspace only accepts destinations from {srgb, lab, xyz, yxy, hsv, cmyk, scrgb, greyscale, lch, ucs}, got '%s'\n",
            arg);
    exit(1);
}

static int run_colourspace(const struct input_blob *input_blob,
                           const VipsInterpretation *targets,
                           int target_count) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    VipsInterpretation route[MAX_COLOURSPACE_STEPS];
    const int effective_steps = target_count > 0
        ? target_count
        : default_colourspace_route(input_blob->interpretation, route);
    const VipsInterpretation *effective_targets = target_count > 0 ? targets : route;
    return run_colourspace_chain_image(in, effective_targets, effective_steps);
}

static int run_srgb_to_lab(const struct input_blob *input_blob) {
    const VipsInterpretation target = VIPS_INTERPRETATION_LAB;
    return run_colourspace(input_blob, &target, 1);
}

static VipsBandFormat parse_cast_target_arg(VipsBandFormat current_format, const char *arg) {
    if (!arg) {
        return current_format == VIPS_FORMAT_UCHAR ? VIPS_FORMAT_FLOAT : VIPS_FORMAT_UCHAR;
    }

    if (strcasecmp(arg, "u8") == 0) return VIPS_FORMAT_UCHAR;
    if (strcasecmp(arg, "u16") == 0) return VIPS_FORMAT_USHORT;
    if (strcasecmp(arg, "f32") == 0) return VIPS_FORMAT_FLOAT;

    fprintf(stderr, "cast only accepts optional target arg 'u8', 'u16', or 'f32', got '%s'\n", arg);
    exit(1);
}

static int run_cast(const struct input_blob *input_blob, VipsBandFormat target) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_cast(in, &out, target, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static VipsDirection parse_flip_direction_arg(const char *arg) {
    if (!arg || strcasecmp(arg, "horizontal") == 0 || strcasecmp(arg, "h") == 0)
        return VIPS_DIRECTION_HORIZONTAL;
    if (strcasecmp(arg, "vertical") == 0 || strcasecmp(arg, "v") == 0)
        return VIPS_DIRECTION_VERTICAL;

    fprintf(stderr, "flip only accepts optional direction arg 'horizontal' or 'vertical', got '%s'\n", arg);
    exit(1);
}

static int run_flip(const struct input_blob *input_blob, VipsDirection direction) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_flip(in, &out, direction, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_gamma(const struct input_blob *input_blob, double exponent) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_gamma(in, &out, "exponent", exponent, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_invert_invert(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *tmp = NULL;
    int ret = vips_invert(in, &tmp, NULL);
    g_object_unref(in);
    if (ret != 0 || !tmp) return -1;

    VipsImage *out = NULL;
    ret = vips_invert(tmp, &out, NULL);
    g_object_unref(tmp);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_invert_invert_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *tmp = NULL;
    int ret = vips_invert(in, &tmp, NULL);
    g_object_unref(in);
    if (ret != 0 || !tmp) return -1;

    VipsImage *out = NULL;
    ret = vips_invert(tmp, &out, NULL);
    g_object_unref(tmp);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static VipsImage *new_rect_mask_image(int kernel_size) {
    const size_t coeff_count = (size_t) kernel_size * kernel_size;
    double *coeff = g_new(double, coeff_count);
    if (!coeff) return NULL;

    for (size_t i = 0; i < coeff_count; i++)
        coeff[i] = 255.0;

    VipsImage *mask = vips_image_new_matrix_from_array(kernel_size,
                                                       kernel_size,
                                                       coeff,
                                                       coeff_count);
    g_free(coeff);
    return mask;
}

static int run_morphology(const struct input_blob *input_blob,
                          int kernel_size,
                          VipsOperationMorphology morph) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *mask = new_rect_mask_image(kernel_size);
    if (!mask) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *out = NULL;
    int ret = vips_morph(in, &out, mask, morph, NULL);
    g_object_unref(mask);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_open(const struct input_blob *input_blob, int kernel_size) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *mask = new_rect_mask_image(kernel_size);
    if (!mask) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *eroded = NULL;
    int ret = vips_morph(in, &eroded, mask, VIPS_OPERATION_MORPHOLOGY_ERODE, NULL);
    g_object_unref(in);
    if (ret != 0 || !eroded) {
        g_object_unref(mask);
        return -1;
    }

    VipsImage *out = NULL;
    ret = vips_morph(eroded, &out, mask, VIPS_OPERATION_MORPHOLOGY_DILATE, NULL);
    g_object_unref(mask);
    g_object_unref(eroded);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_close(const struct input_blob *input_blob, int kernel_size) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *mask = new_rect_mask_image(kernel_size);
    if (!mask) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *dilated = NULL;
    int ret = vips_morph(in, &dilated, mask, VIPS_OPERATION_MORPHOLOGY_DILATE, NULL);
    g_object_unref(in);
    if (ret != 0 || !dilated) {
        g_object_unref(mask);
        return -1;
    }

    VipsImage *out = NULL;
    ret = vips_morph(dilated, &out, mask, VIPS_OPERATION_MORPHOLOGY_ERODE, NULL);
    g_object_unref(mask);
    g_object_unref(dilated);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_sharpen(const struct input_blob *input_blob, double sigma, double strength) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_sharpen(in, &out,
                           "sigma", sigma,
                           "x1", 2.0,
                           "y2", 10.0,
                           "y3", 20.0,
                           "m1", 0.0,
                           "m2", strength,
                           NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_unsharp_mask(const struct input_blob *input_blob, double sigma, double strength) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *blur = NULL;
    if (vips_gaussblur(in, &blur, sigma, NULL) != 0 || !blur) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *diff = NULL;
    if (vips_subtract(in, blur, &diff, NULL) != 0 || !diff) {
        g_object_unref(blur);
        g_object_unref(in);
        return -1;
    }

    VipsImage *scaled = NULL;
    if (vips_linear1(diff, &scaled, strength, 0.0, NULL) != 0 || !scaled) {
        g_object_unref(diff);
        g_object_unref(blur);
        g_object_unref(in);
        return -1;
    }

    VipsImage *out = NULL;
    int ret = vips_add(in, scaled, &out, NULL);
    g_object_unref(scaled);
    g_object_unref(diff);
    g_object_unref(blur);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_thumbnail_sharpen(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *thumb = NULL;
    int ret = vips_thumbnail_image(in, &thumb, 400, NULL);
    g_object_unref(in);
    if (ret != 0 || !thumb) return -1;

    VipsImage *sharp = NULL;
    ret = vips_sharpen(thumb, &sharp,
                       "sigma", 0.5,
                       "x1", 2.0,
                       "y2", 10.0,
                       "y3", 20.0,
                       "m1", 0.0,
                       "m2", 3.0,
                       NULL);
    g_object_unref(thumb);
    if (ret != 0 || !sharp) return -1;

    int sink_ret = vips_sink_discard(sharp);
    g_object_unref(sharp);
    return sink_ret;
}

static int run_thumbnail_gauss_blur(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *thumb = NULL;
    int ret = vips_thumbnail_image(in, &thumb, 400, NULL);
    g_object_unref(in);
    if (ret != 0 || !thumb) return -1;

    VipsImage *out = NULL;
    ret = vips_gaussblur(thumb, &out, 2.0, NULL);
    g_object_unref(thumb);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_thumbnail_linear(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *thumb = NULL;
    int ret = vips_thumbnail_image(in, &thumb, 400, NULL);
    g_object_unref(in);
    if (ret != 0 || !thumb) return -1;

    VipsImage *out = NULL;
    ret = vips_linear1(thumb, &out, 1.2, 0.0, NULL);
    g_object_unref(thumb);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_thumbnail_colourspace_cast(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *thumb = NULL;
    int ret = vips_thumbnail_image(in, &thumb, 400, NULL);
    g_object_unref(in);
    if (ret != 0 || !thumb) return -1;

    VipsImage *lab = NULL;
    ret = vips_colourspace(thumb, &lab, VIPS_INTERPRETATION_LAB, NULL);
    g_object_unref(thumb);
    if (ret != 0 || !lab) return -1;

    VipsImage *out = NULL;
    ret = vips_cast(lab, &out, VIPS_FORMAT_UCHAR, NULL);
    g_object_unref(lab);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_resize_colourspace(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *resized = NULL;
    int ret = vips_resize(in, &resized, 0.5, NULL);
    g_object_unref(in);
    if (ret != 0 || !resized) return -1;

    VipsImage *out = NULL;
    ret = vips_colourspace(resized, &out, VIPS_INTERPRETATION_LAB, NULL);
    g_object_unref(resized);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_embed(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    const int src_width = vips_image_get_width(in);
    const int src_height = vips_image_get_height(in);
    VipsImage *out = NULL;
    int ret = vips_embed(in, &out,
                         EMBED_OFFSET_X, EMBED_OFFSET_Y,
                         src_width + EMBED_PAD_WIDTH,
                         src_height + EMBED_PAD_HEIGHT,
                         "extend", VIPS_EXTEND_COPY,
                         NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_extract_area(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    const int src_width = vips_image_get_width(in);
    const int src_height = vips_image_get_height(in);
    VipsImage *out = NULL;
    int ret = vips_extract_area(in, &out,
                                EXTRACT_OFFSET_X, EXTRACT_OFFSET_Y,
                                src_width - EXTRACT_TRIM_WIDTH,
                                src_height - EXTRACT_TRIM_HEIGHT,
                                NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_embed_extract(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    const int src_width = vips_image_get_width(in);
    const int src_height = vips_image_get_height(in);
    const int canvas_width = src_width > 2048 ? src_width : 2048;
    const int canvas_height = src_height > 2048 ? src_height : 2048;

    VipsImage *embedded = NULL;
    int ret = vips_embed(in, &embedded, 0, 0, canvas_width, canvas_height, "extend", VIPS_EXTEND_COPY, NULL);
    g_object_unref(in);
    if (ret != 0 || !embedded) return -1;

    VipsImage *out = NULL;
    ret = vips_extract_area(embedded, &out, 100, 100, 800, 600, NULL);
    g_object_unref(embedded);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_three_op_chain(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *thumb = NULL;
    int ret = vips_thumbnail_image(in, &thumb, 400, NULL);
    g_object_unref(in);
    if (ret != 0 || !thumb) return -1;

    VipsImage *sharp = NULL;
    ret = vips_sharpen(thumb, &sharp,
                       "sigma", 0.5,
                       "x1", 2.0,
                       "y2", 10.0,
                       "y3", 20.0,
                       "m1", 0.0,
                       "m2", 3.0,
                       NULL);
    g_object_unref(thumb);
    if (ret != 0 || !sharp) return -1;

    VipsImage *out = NULL;
    ret = vips_gaussblur(sharp, &out, 1.0, NULL);
    g_object_unref(sharp);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static VipsImage *new_conv_mask_image(const double *coeff,
                                      int kernel_size,
                                      double scale,
                                      double offset) {
    const size_t coeff_count = (size_t) kernel_size * kernel_size;
    VipsImage *mask = vips_image_new_matrix_from_array(kernel_size,
                                                       kernel_size,
                                                       coeff,
                                                       coeff_count);
    if (!mask) return NULL;
    vips_image_set_double(mask, "scale", scale);
    vips_image_set_double(mask, "offset", offset);
    return mask;
}

static int run_conv(const struct input_blob *input_blob,
                    const double *coeff,
                    int kernel_size,
                    double scale,
                    double offset) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *mask = new_conv_mask_image(coeff, kernel_size, scale, offset);
    if (!mask) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *out = NULL;
    int ret = vips_conv(in, &out, mask, NULL);
    g_object_unref(mask);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_sobel(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_sobel(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_prewitt(const struct input_blob *input_blob) {
#if VIPS_MAJOR_VERSION > 8 || (VIPS_MAJOR_VERSION == 8 && VIPS_MINOR_VERSION >= 16)
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_prewitt(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
#else
    (void)input_blob;
    fprintf(stderr, "vips_prewitt requires libvips >= 8.16\n");
    return -1;
#endif
}

static int run_median_blur(const struct input_blob *input_blob, int size) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_median(in, &out, size, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_load(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    void *buf = vips_image_write_to_memory(in, NULL);
    g_object_unref(in);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_load_jpeg(const char *input) {
    VipsImage *in = NULL;
    if (vips_jpegload(input, &in, NULL) != 0 || !in) return -1;

    void *buf = vips_image_write_to_memory(in, NULL);
    g_object_unref(in);
    g_free(buf);
    return buf ? 0 : -1;
}

typedef int (*op_fn)(const char *input, void *ctx);

struct thumbnail_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    int width;
};
struct resize_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    double scale;
};
struct zoom_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    int xfac;
    int yfac;
};
struct affine_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    double a;
    double b;
    double c;
    double d;
};
struct similarity_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    double scale;
    double angle;
};
struct shrink_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    int factor;
};
struct composite_shrink_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    int hfactor;
    int vfactor;
};
struct linear_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    double scale;
    double offset;
};
struct cast_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    VipsBandFormat target;
};
struct composite_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    VipsBlendMode mode;
};
struct flip_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    VipsDirection direction;
};
struct gauss_blur_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    double sigma;
};
struct gamma_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    double exponent;
};
struct colourspace_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    VipsInterpretation targets[MAX_COLOURSPACE_STEPS];
    int target_count;
};
struct invert_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
};
struct bandmean_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
};
struct chain_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
};
struct sharpen_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    double sigma;
    double strength;
};
struct conv_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    const double *coeff;
    int kernel_size;
    double scale;
    double offset;
};
struct morphology_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    int kernel_size;
};
struct mapim_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    float *index_buf;
    int width;
    int height;
    double dx;
    double dy;
};
struct save_avif_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
};
struct save_exr_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
};
struct save_gif_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
};
struct save_heif_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
};
struct save_jpeg_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
};
struct save_jp2k_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
};
struct save_tiff_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    VipsForeignTiffCompression compression;
};
struct workflow_ctx {
    const struct input_blob *input_blob; /* NULL in e2e mode */
    int width;
    const char *target_format; /* ".jpg", ".webp", ".avif", ".png", ".tif" */
};

static VipsForeignTiffCompression parse_tiff_compression_arg(const char *arg) {
    if (!arg || strcmp(arg, "none") == 0)
        return VIPS_FOREIGN_TIFF_COMPRESSION_NONE;
    if (strcmp(arg, "lzw") == 0)
        return VIPS_FOREIGN_TIFF_COMPRESSION_LZW;
    if (strcmp(arg, "deflate") == 0)
        return VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE;
    if (strcmp(arg, "packbits") == 0)
        return VIPS_FOREIGN_TIFF_COMPRESSION_PACKBITS;
    if (strcmp(arg, "jpeg") == 0)
        return VIPS_FOREIGN_TIFF_COMPRESSION_JPEG;

    fprintf(stderr, "save-tiff only accepts optional compression arg 'none', 'lzw', 'deflate', 'packbits', or 'jpeg', got '%s'\n", arg);
    exit(1);
}

/*
 * E2E helpers: decode from file on every call so the iteration includes
 * full codec decode cost (the productive pipeline scenario).
 */
static int run_thumbnail_e2e(const char *input, int width) {
    /* vips_thumbnail operates directly on the file, enabling shrink-on-load */
    VipsImage *out = NULL;
    int ret = vips_thumbnail(input, &out, width, NULL);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_resize_e2e(const char *input, double scale) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_resize(in, &out, scale, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_zoom_e2e(const char *input, int xfac, int yfac) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_zoom(in, &out, xfac, yfac, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_affine_e2e(const char *input,
                          double a, double b, double c, double d) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    int width = vips_image_get_width(in);
    int height = vips_image_get_height(in);
    VipsArea *oarea = VIPS_AREA(vips_array_int_newv(4, 0, 0, width, height));
    VipsImage *out = NULL;
    int ret = vips_affine(in, &out, a, b, c, d, "oarea", oarea, NULL);
    vips_area_unref(oarea);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_similarity_e2e(const char *input, double scale, double angle) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_similarity(in, &out, "scale", scale, "angle", angle, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_shrinkh_e2e(const char *input, int factor) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_shrinkh(in, &out, factor, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_linear_e2e(const char *input, double scale, double offset) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_linear1(in, &out, scale, offset, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_add_e2e(const char *input) {
    return run_linear_e2e(input, 1.0, 5.0);
}

static int run_multiply_e2e(const char *input) {
    return run_linear_e2e(input, 2.0, 0.0);
}

static int run_subtract_e2e(const char *input) {
    return run_linear_e2e(input, 1.0, -5.0);
}

static int run_and_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_andimage_const1(in, &out, 240.0, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_equal_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    double rhs = 128.0;
    if (vips_image_get_format(in) == VIPS_FORMAT_USHORT)
        rhs = 32768.0;
    else if (vips_image_get_format(in) == VIPS_FORMAT_FLOAT
          || vips_image_get_format(in) == VIPS_FORMAT_DOUBLE)
        rhs = 0.5;

    VipsImage *out = NULL;
    int ret = vips_equal_const1(in, &out, rhs, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_histogram_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_hist_find(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_recomb_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    if (vips_image_get_bands(in) != 3) {
        fprintf(stderr, "recomb benchmark expects a 3-band input image, got %d\n", vips_image_get_bands(in));
        g_object_unref(in);
        return -1;
    }

    static const double matrix_values[6] = {
        0.299, 0.587, 0.114,
        1.000, 0.000, -1.000
    };
    VipsImage *matrix = vips_image_new_matrix_from_array(3, 2, matrix_values, 6);
    if (!matrix) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *out = NULL;
    int ret = vips_recomb(in, &out, matrix, NULL);
    g_object_unref(matrix);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_grey_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    const int width = vips_image_get_width(in);
    const int height = vips_image_get_height(in);
    g_object_unref(in);
    return run_grey(width, height);
}

static int run_draw_line_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    const int width = vips_image_get_width(in);
    const int height = vips_image_get_height(in);
    const int bands = vips_image_get_bands(in);
    g_object_unref(in);
    return run_draw_line(width, height, bands);
}

static int run_draw_rect_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    const int width = vips_image_get_width(in);
    const int height = vips_image_get_height(in);
    const int bands = vips_image_get_bands(in);
    g_object_unref(in);
    return run_draw_rect(width, height, bands);
}

static int run_draw_circle_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    const int width = vips_image_get_width(in);
    const int height = vips_image_get_height(in);
    const int bands = vips_image_get_bands(in);
    g_object_unref(in);
    return run_draw_circle(width, height, bands);
}

static int run_freqfilt_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *mono = NULL;
    int ret = vips_image_get_bands(in) > 1
        ? vips_bandmean(in, &mono, NULL)
        : vips_copy(in, &mono, NULL);
    g_object_unref(in);
    if (ret != 0 || !mono) return -1;

    VipsImage *fft = NULL;
    ret = vips_fwfft(mono, &fft, NULL);
    g_object_unref(mono);
    if (ret != 0 || !fft) return -1;

    VipsImage *out = NULL;
    ret = vips_invfft(fft, &out, "real", TRUE, NULL);
    g_object_unref(fft);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_shrinkv_e2e(const char *input, int factor) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_shrinkv(in, &out, factor, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_shrink_e2e(const char *input, int hfactor, int vfactor) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_shrink(in, &out, hfactor, vfactor, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_gauss_blur_e2e(const char *input, double sigma) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_gaussblur(in, &out, sigma, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_colourspace_e2e(const char *input,
                               const VipsInterpretation *targets,
                               int target_count) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsInterpretation route[MAX_COLOURSPACE_STEPS];
    const int effective_steps = target_count > 0
        ? target_count
        : default_colourspace_route(in->Type, route);
    const VipsInterpretation *effective_targets = target_count > 0 ? targets : route;
    return run_colourspace_chain_image(in, effective_targets, effective_steps);
}

static int run_srgb_to_lab_e2e(const char *input) {
    const VipsInterpretation target = VIPS_INTERPRETATION_LAB;
    return run_colourspace_e2e(input, &target, 1);
}

static int run_cast_e2e(const char *input, VipsBandFormat target) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_cast(in, &out, target, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_flip_e2e(const char *input, VipsDirection direction) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_flip(in, &out, direction, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_gamma_e2e(const char *input, double exponent) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_gamma(in, &out, "exponent", exponent, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_thumbnail_sharpen_e2e(const char *input) {
    VipsImage *thumb = NULL;
    int ret = vips_thumbnail(input, &thumb, 400, NULL);
    if (ret != 0 || !thumb) return -1;

    VipsImage *sharp = NULL;
    ret = vips_sharpen(thumb, &sharp,
                       "sigma", 0.5,
                       "x1", 2.0,
                       "y2", 10.0,
                       "y3", 20.0,
                       "m1", 0.0,
                       "m2", 3.0,
                       NULL);
    g_object_unref(thumb);
    if (ret != 0 || !sharp) return -1;

    void *buf = vips_image_write_to_memory(sharp, NULL);
    g_object_unref(sharp);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_thumbnail_gauss_blur_e2e(const char *input) {
    VipsImage *thumb = NULL;
    int ret = vips_thumbnail(input, &thumb, 400, NULL);
    if (ret != 0 || !thumb) return -1;

    VipsImage *out = NULL;
    ret = vips_gaussblur(thumb, &out, 2.0, NULL);
    g_object_unref(thumb);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_thumbnail_linear_e2e(const char *input) {
    VipsImage *thumb = NULL;
    int ret = vips_thumbnail(input, &thumb, 400, NULL);
    if (ret != 0 || !thumb) return -1;

    VipsImage *out = NULL;
    ret = vips_linear1(thumb, &out, 1.2, 0.0, NULL);
    g_object_unref(thumb);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_thumbnail_colourspace_cast_e2e(const char *input) {
    VipsImage *thumb = NULL;
    int ret = vips_thumbnail(input, &thumb, 400, NULL);
    if (ret != 0 || !thumb) return -1;

    VipsImage *lab = NULL;
    ret = vips_colourspace(thumb, &lab, VIPS_INTERPRETATION_LAB, NULL);
    g_object_unref(thumb);
    if (ret != 0 || !lab) return -1;

    VipsImage *out = NULL;
    ret = vips_cast(lab, &out, VIPS_FORMAT_UCHAR, NULL);
    g_object_unref(lab);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_resize_colourspace_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *resized = NULL;
    int ret = vips_resize(in, &resized, 0.5, NULL);
    g_object_unref(in);
    if (ret != 0 || !resized) return -1;

    VipsImage *out = NULL;
    ret = vips_colourspace(resized, &out, VIPS_INTERPRETATION_LAB, NULL);
    g_object_unref(resized);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_embed_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    const int src_width = vips_image_get_width(in);
    const int src_height = vips_image_get_height(in);
    VipsImage *out = NULL;
    int ret = vips_embed(in, &out,
                         EMBED_OFFSET_X, EMBED_OFFSET_Y,
                         src_width + EMBED_PAD_WIDTH,
                         src_height + EMBED_PAD_HEIGHT,
                         "extend", VIPS_EXTEND_COPY,
                         NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_extract_area_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    const int src_width = vips_image_get_width(in);
    const int src_height = vips_image_get_height(in);
    VipsImage *out = NULL;
    int ret = vips_extract_area(in, &out,
                                EXTRACT_OFFSET_X, EXTRACT_OFFSET_Y,
                                src_width - EXTRACT_TRIM_WIDTH,
                                src_height - EXTRACT_TRIM_HEIGHT,
                                NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_embed_extract_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    const int src_width = vips_image_get_width(in);
    const int src_height = vips_image_get_height(in);
    const int canvas_width = src_width > 2048 ? src_width : 2048;
    const int canvas_height = src_height > 2048 ? src_height : 2048;

    VipsImage *embedded = NULL;
    int ret = vips_embed(in, &embedded, 0, 0, canvas_width, canvas_height, "extend", VIPS_EXTEND_COPY, NULL);
    g_object_unref(in);
    if (ret != 0 || !embedded) return -1;

    VipsImage *out = NULL;
    ret = vips_extract_area(embedded, &out, 100, 100, 800, 600, NULL);
    g_object_unref(embedded);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_three_op_chain_e2e(const char *input) {
    VipsImage *thumb = NULL;
    int ret = vips_thumbnail(input, &thumb, 400, NULL);
    if (ret != 0 || !thumb) return -1;

    VipsImage *sharp = NULL;
    ret = vips_sharpen(thumb, &sharp,
                       "sigma", 0.5,
                       "x1", 2.0,
                       "y2", 10.0,
                       "y3", 20.0,
                       "m1", 0.0,
                       "m2", 3.0,
                       NULL);
    g_object_unref(thumb);
    if (ret != 0 || !sharp) return -1;

    VipsImage *out = NULL;
    ret = vips_gaussblur(sharp, &out, 1.0, NULL);
    g_object_unref(sharp);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}
/*
 * perceptual_enhance: production e-commerce image pipeline.
 *
 * thumbnail(800px, Lanczos3)
 *   → sRGB → Lab             (perceptual space: contrast without hue shift)
 *   → linear(×1.05, −3.5)    (contrast lift on L*, a*, b*)
 *   → Lab → sRGB
 *   → sharpen(σ=0.5)         (unsharp mask; vips_sharpen converts to Lab internally)
 *   → gamma(0.95)            (compensate slight darkening from sharpening)
 *   → WebP encode
 */
static int run_perceptual_enhance(const struct input_blob *input_blob,
                                  int width,
                                  const char *target_format) {
    (void)target_format;
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *thumb = NULL;
    int ret = vips_thumbnail_image(in, &thumb, width, NULL);
    g_object_unref(in);
    if (ret != 0 || !thumb) return -1;

    /* sRGB → Lab */
    VipsImage *lab = NULL;
    ret = vips_colourspace(thumb, &lab, VIPS_INTERPRETATION_LAB, NULL);
    g_object_unref(thumb);
    if (ret != 0 || !lab) return -1;

    /* linear contrast lift: ×1.05, offset −3.5 on all bands */
    double scale[3]  = {1.05, 1.05, 1.05};
    double offset[3] = {-3.5, -3.5, -3.5};
    VipsImage *lifted = NULL;
    ret = vips_linear(lab, &lifted, scale, offset, 3, NULL);
    g_object_unref(lab);
    if (ret != 0 || !lifted) return -1;

    /* Lab → sRGB */
    VipsImage *srgb = NULL;
    ret = vips_colourspace(lifted, &srgb, VIPS_INTERPRETATION_sRGB, NULL);
    g_object_unref(lifted);
    if (ret != 0 || !srgb) return -1;

    /* unsharp mask (vips_sharpen internally converts to Lab) */
    VipsImage *sharp = NULL;
    ret = vips_sharpen(srgb, &sharp,
                       "sigma", 0.5,
                       "x1", 2.0,
                       "y2", 10.0,
                       "y3", 20.0,
                       "m1", 0.0,
                       "m2", 3.0,
                       NULL);
    g_object_unref(srgb);
    if (ret != 0 || !sharp) return -1;

    /* gamma(0.95) */
    VipsImage *gammad = NULL;
    ret = vips_gamma(sharp, &gammad, "exponent", 0.95, NULL);
    g_object_unref(sharp);
    if (ret != 0 || !gammad) return -1;

    ret = vips_sink_discard(gammad);
    g_object_unref(gammad);
    return ret;
}

static int run_perceptual_enhance_e2e(const char *input,
                                      int width,
                                      const char *target_format) {
    VipsImage *thumb = NULL;
    int ret = vips_thumbnail(input, &thumb, width, NULL);
    if (ret != 0 || !thumb) return -1;

    VipsImage *lab = NULL;
    ret = vips_colourspace(thumb, &lab, VIPS_INTERPRETATION_LAB, NULL);
    g_object_unref(thumb);
    if (ret != 0 || !lab) return -1;

    double scale[3]  = {1.05, 1.05, 1.05};
    double offset[3] = {-3.5, -3.5, -3.5};
    VipsImage *lifted = NULL;
    ret = vips_linear(lab, &lifted, scale, offset, 3, NULL);
    g_object_unref(lab);
    if (ret != 0 || !lifted) return -1;

    VipsImage *srgb = NULL;
    ret = vips_colourspace(lifted, &srgb, VIPS_INTERPRETATION_sRGB, NULL);
    g_object_unref(lifted);
    if (ret != 0 || !srgb) return -1;

    VipsImage *sharp = NULL;
    ret = vips_sharpen(srgb, &sharp,
                       "sigma", 0.5,
                       "x1", 2.0,
                       "y2", 10.0,
                       "y3", 20.0,
                       "m1", 0.0,
                       "m2", 3.0,
                       NULL);
    g_object_unref(srgb);
    if (ret != 0 || !sharp) return -1;

    VipsImage *gammad = NULL;
    ret = vips_gamma(sharp, &gammad, "exponent", 0.95, NULL);
    g_object_unref(sharp);
    if (ret != 0 || !gammad) return -1;

    void *buf = NULL;
    size_t len = 0;
    ret = vips_image_write_to_buffer(gammad, target_format, &buf, &len, NULL);
    g_object_unref(gammad);
    if (ret != 0 || !buf || len == 0) {
        if (buf) g_free(buf);
        return -1;
    }
    g_free(buf);
    return 0;
}

static int op_perceptual_enhance(const char *input, void *ctx) {
    struct workflow_ctx *wf = (struct workflow_ctx *)ctx;
    if (!wf->input_blob)
        return run_perceptual_enhance_e2e(input, wf->width, wf->target_format);
    return run_perceptual_enhance(wf->input_blob, wf->width, wf->target_format);
}

static int run_save_avif_image(VipsImage *in) {
    void *buf = NULL;
    size_t len = 0;
    int ret = vips_image_write_to_buffer(in, ".avif", &buf, &len, NULL);
    if (ret != 0 || !buf || len == 0) {
        if (buf) g_free(buf);
        return -1;
    }
    g_free(buf);
    return 0;
}

static int run_save_avif(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    int ret = run_save_avif_image(in);
    g_object_unref(in);
    return ret;
}

static int run_save_gif_image(VipsImage *in) {
    void *buf = NULL;
    size_t len = 0;
    int ret = vips_image_write_to_buffer(in, ".gif", &buf, &len, NULL);
    if (ret != 0 || !buf || len == 0) {
        if (buf) g_free(buf);
        return -1;
    }
    g_free(buf);
    return 0;
}

static int run_save_gif(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    int ret = run_save_gif_image(in);
    g_object_unref(in);
    return ret;
}

static int run_save_heif_image(VipsImage *in) {
    void *buf = NULL;
    size_t len = 0;
    int ret = vips_image_write_to_buffer(in, ".heic", &buf, &len, NULL);
    if (ret != 0 || !buf || len == 0) {
        if (buf) g_free(buf);
        return -1;
    }
    g_free(buf);
    return 0;
}

static int run_save_heif(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    int ret = run_save_heif_image(in);
    g_object_unref(in);
    return ret;
}

static int run_save_jpeg_image(VipsImage *in) {
    void *buf = NULL;
    size_t len = 0;
    int ret = vips_image_write_to_buffer(in, ".jpg", &buf, &len,
                                         "Q", 85,
                                         NULL);
    if (ret != 0 || !buf || len == 0) {
        if (buf) g_free(buf);
        return -1;
    }
    g_free(buf);
    return 0;
}

static int run_save_jpeg(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    int ret = run_save_jpeg_image(in);
    g_object_unref(in);
    return ret;
}

static int run_save_jp2k_image(VipsImage *in) {
    void *buf = NULL;
    size_t len = 0;
    int ret = vips_image_write_to_buffer(in, ".jp2", &buf, &len, NULL);
    if (ret != 0 || !buf || len == 0) {
        if (buf) g_free(buf);
        return -1;
    }
    g_free(buf);
    return 0;
}

static int run_save_tiff_image(VipsImage *in, VipsForeignTiffCompression compression) {
    void *buf = NULL;
    size_t len = 0;
    int ret = vips_image_write_to_buffer(in, ".tif", &buf, &len,
                                         "compression", compression,
                                         NULL);
    if (ret != 0 || !buf || len == 0) {
        if (buf) g_free(buf);
        return -1;
    }
    g_free(buf);
    return 0;
}

static int run_save_exr_image(VipsImage *in) {
    void *buf = NULL;
    size_t len = 0;
    int ret = vips_image_write_to_buffer(in, ".exr", &buf, &len, NULL);
    if (ret != 0 || !buf || len == 0) {
        if (buf) g_free(buf);
        return -1;
    }
    g_free(buf);
    return 0;
}

static int run_save_jp2k(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    int ret = run_save_jp2k_image(in);
    g_object_unref(in);
    return ret;
}

static int run_save_exr(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    int ret = run_save_exr_image(in);
    g_object_unref(in);
    return ret;
}

static int run_save_tiff(const struct input_blob *input_blob, VipsForeignTiffCompression compression) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    int ret = run_save_tiff_image(in, compression);
    g_object_unref(in);
    return ret;
}

static int run_invert_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_invert(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_abs(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_abs(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_abs_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_abs(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_sign(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_sign(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_sign_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_sign(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_round(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_rint(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_round_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_rint(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_floor_op(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_floor(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_floor_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_floor(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_ceil_op(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_ceil(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_ceil_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_ceil(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_bandmean(const struct input_blob *input_blob) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_bandmean(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int run_bandmean_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_bandmean(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_mapim_e2e(const char *input, const float *index_buf, int width, int height) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *index = vips_image_new_from_memory(index_buf,
                                                  (size_t) width * height * 2 * sizeof(float),
                                                  width,
                                                  height,
                                                  2,
                                                  VIPS_FORMAT_FLOAT);
    if (!index) {
        g_object_unref(in);
        return -1;
    }
    VipsImage *out = NULL;
    int ret = vips_mapim(in, &out, index, "extend", VIPS_EXTEND_COPY, NULL);
    g_object_unref(index);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int prepare_mapim_index(struct mapim_ctx *ctx) {
    size_t sample_count = (size_t) ctx->width * ctx->height * 2;
    ctx->index_buf = g_malloc(sizeof(float) * sample_count);
    if (!ctx->index_buf)
        return -1;

    for (int y = 0; y < ctx->height; y++) {
        for (int x = 0; x < ctx->width; x++) {
            size_t base = ((size_t) y * ctx->width + x) * 2;
            ctx->index_buf[base] = (float) x + (float) ctx->dx;
            ctx->index_buf[base + 1] = (float) y + (float) ctx->dy;
        }
    }

    return 0;
}

static int run_sharpen_e2e(const char *input, double sigma, double strength) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_sharpen(in, &out,
                           "sigma", sigma,
                           "x1", 2.0,
                           "y2", 10.0,
                           "y3", 20.0,
                           "m1", 0.0,
                           "m2", strength,
                           NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_unsharp_mask_e2e(const char *input, double sigma, double strength) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *blur = NULL;
    if (vips_gaussblur(in, &blur, sigma, NULL) != 0 || !blur) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *diff = NULL;
    if (vips_subtract(in, blur, &diff, NULL) != 0 || !diff) {
        g_object_unref(blur);
        g_object_unref(in);
        return -1;
    }

    VipsImage *scaled = NULL;
    if (vips_linear1(diff, &scaled, strength, 0.0, NULL) != 0 || !scaled) {
        g_object_unref(diff);
        g_object_unref(blur);
        g_object_unref(in);
        return -1;
    }

    VipsImage *out = NULL;
    int ret = vips_add(in, scaled, &out, NULL);
    g_object_unref(scaled);
    g_object_unref(diff);
    g_object_unref(blur);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_conv_e2e(const char *input,
                        const double *coeff,
                        int kernel_size,
                        double scale,
                        double offset) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *mask = new_conv_mask_image(coeff, kernel_size, scale, offset);
    if (!mask) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *out = NULL;
    int ret = vips_conv(in, &out, mask, NULL);
    g_object_unref(mask);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_sobel_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_sobel(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_prewitt_e2e(const char *input) {
#if VIPS_MAJOR_VERSION > 8 || (VIPS_MAJOR_VERSION == 8 && VIPS_MINOR_VERSION >= 16)
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_prewitt(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
#else
    (void)input;
    fprintf(stderr, "vips_prewitt requires libvips >= 8.16\n");
    return -1;
#endif
}

static int run_median_blur_e2e(const char *input, int size) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *out = NULL;
    int ret = vips_median(in, &out, size, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_morphology_e2e(const char *input,
                              int kernel_size,
                              VipsOperationMorphology morph) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *mask = new_rect_mask_image(kernel_size);
    if (!mask) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *out = NULL;
    int ret = vips_morph(in, &out, mask, morph, NULL);
    g_object_unref(mask);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_open_e2e(const char *input, int kernel_size) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *mask = new_rect_mask_image(kernel_size);
    if (!mask) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *eroded = NULL;
    int ret = vips_morph(in, &eroded, mask, VIPS_OPERATION_MORPHOLOGY_ERODE, NULL);
    g_object_unref(in);
    if (ret != 0 || !eroded) {
        g_object_unref(mask);
        return -1;
    }

    VipsImage *out = NULL;
    ret = vips_morph(eroded, &out, mask, VIPS_OPERATION_MORPHOLOGY_DILATE, NULL);
    g_object_unref(mask);
    g_object_unref(eroded);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_close_e2e(const char *input, int kernel_size) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;

    VipsImage *mask = new_rect_mask_image(kernel_size);
    if (!mask) {
        g_object_unref(in);
        return -1;
    }

    VipsImage *dilated = NULL;
    int ret = vips_morph(in, &dilated, mask, VIPS_OPERATION_MORPHOLOGY_DILATE, NULL);
    g_object_unref(in);
    if (ret != 0 || !dilated) {
        g_object_unref(mask);
        return -1;
    }

    VipsImage *out = NULL;
    ret = vips_morph(dilated, &out, mask, VIPS_OPERATION_MORPHOLOGY_ERODE, NULL);
    g_object_unref(mask);
    g_object_unref(dilated);
    if (ret != 0 || !out) return -1;

    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}

static int run_save_avif_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    int ret = run_save_avif_image(in);
    g_object_unref(in);
    return ret;
}

static int run_save_gif_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    int ret = run_save_gif_image(in);
    g_object_unref(in);
    return ret;
}

static int run_save_heif_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    int ret = run_save_heif_image(in);
    g_object_unref(in);
    return ret;
}

static int run_save_jpeg_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    int ret = run_save_jpeg_image(in);
    g_object_unref(in);
    return ret;
}

static int run_save_jp2k_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    int ret = run_save_jp2k_image(in);
    g_object_unref(in);
    return ret;
}

static int run_save_exr_e2e(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    int ret = run_save_exr_image(in);
    g_object_unref(in);
    return ret;
}

static int run_save_tiff_e2e(const char *input, VipsForeignTiffCompression compression) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    int ret = run_save_tiff_image(in, compression);
    g_object_unref(in);
    return ret;
}

/*
 * Production workflow: decode → thumbnail → sharpen → encode to target format.
 * Simulates a web server processing images on-the-fly.
 */
static int run_workflow(const struct input_blob *input_blob, int width,
                        const char *target_format) {
    VipsImage *in = new_input_image(input_blob);
    if (!in) return -1;

    VipsImage *thumb = NULL;
    int ret = vips_thumbnail_image(in, &thumb, width, NULL);
    g_object_unref(in);
    if (ret != 0 || !thumb) return -1;

    VipsImage *sharp = NULL;
    ret = vips_sharpen(thumb, &sharp,
                       "sigma", 0.5,
                       "x1", 2.0,
                       "y2", 10.0,
                       "y3", 20.0,
                       "m1", 0.0,
                       "m2", 3.0,
                       NULL);
    g_object_unref(thumb);
    if (ret != 0 || !sharp) return -1;

    void *buf = NULL;
    size_t len = 0;
    ret = vips_image_write_to_buffer(sharp, target_format, &buf, &len, NULL);
    g_object_unref(sharp);
    if (ret != 0 || !buf || len == 0) {
        if (buf) g_free(buf);
        return -1;
    }
    g_free(buf);
    return 0;
}

static int run_workflow_e2e(const char *input, int width,
                            const char *target_format) {
    VipsImage *out = NULL;
    int ret = vips_thumbnail(input, &out, width, NULL);
    if (ret != 0 || !out) return -1;

    VipsImage *sharp = NULL;
    ret = vips_sharpen(out, &sharp,
                       "sigma", 0.5,
                       "x1", 2.0,
                       "y2", 10.0,
                       "y3", 20.0,
                       "m1", 0.0,
                       "m2", 3.0,
                       NULL);
    g_object_unref(out);
    if (ret != 0 || !sharp) return -1;

    void *buf = NULL;
    size_t len = 0;
    ret = vips_image_write_to_buffer(sharp, target_format, &buf, &len, NULL);
    g_object_unref(sharp);
    if (ret != 0 || !buf || len == 0) {
        if (buf) g_free(buf);
        return -1;
    }
    g_free(buf);
    return 0;
}

static int op_thumbnail(const char *input, void *ctx) {
    struct thumbnail_ctx *thumbnail = (struct thumbnail_ctx *)ctx;
    if (!thumbnail->input_blob)
        return run_thumbnail_e2e(input, thumbnail->width);
    return run_thumbnail(thumbnail->input_blob, thumbnail->width);
}

static int op_resize(const char *input, void *ctx) {
    struct resize_ctx *resize = (struct resize_ctx *)ctx;
    if (!resize->input_blob)
        return run_resize_e2e(input, resize->scale);
    return run_resize(resize->input_blob, resize->scale);
}

static int op_zoom(const char *input, void *ctx) {
    struct zoom_ctx *zoom = (struct zoom_ctx *)ctx;
    if (!zoom->input_blob)
        return run_zoom_e2e(input, zoom->xfac, zoom->yfac);
    return run_zoom(zoom->input_blob, zoom->xfac, zoom->yfac);
}

static int op_affine(const char *input, void *ctx) {
    struct affine_ctx *affine = (struct affine_ctx *)ctx;
    if (!affine->input_blob)
        return run_affine_e2e(input, affine->a, affine->b, affine->c, affine->d);
    return run_affine(affine->input_blob, affine->a, affine->b, affine->c, affine->d);
}

static int op_similarity(const char *input, void *ctx) {
    struct similarity_ctx *similarity = (struct similarity_ctx *)ctx;
    if (!similarity->input_blob)
        return run_similarity_e2e(input, similarity->scale, similarity->angle);
    return run_similarity(similarity->input_blob, similarity->scale, similarity->angle);
}

static int op_shrinkh(const char *input, void *ctx) {
    struct shrink_ctx *shrink = (struct shrink_ctx *)ctx;
    if (!shrink->input_blob)
        return run_shrinkh_e2e(input, shrink->factor);
    return run_shrinkh(shrink->input_blob, shrink->factor);
}

static int op_shrinkv(const char *input, void *ctx) {
    struct shrink_ctx *shrink = (struct shrink_ctx *)ctx;
    if (!shrink->input_blob)
        return run_shrinkv_e2e(input, shrink->factor);
    return run_shrinkv(shrink->input_blob, shrink->factor);
}

static int op_shrink(const char *input, void *ctx) {
    struct composite_shrink_ctx *shrink = (struct composite_shrink_ctx *)ctx;
    if (!shrink->input_blob)
        return run_shrink_e2e(input, shrink->hfactor, shrink->vfactor);
    return run_shrink(shrink->input_blob, shrink->hfactor, shrink->vfactor);
}

static int op_linear(const char *input, void *ctx) {
    struct linear_ctx *linear = (struct linear_ctx *)ctx;
    if (!linear->input_blob)
        return run_linear_e2e(input, linear->scale, linear->offset);
    return run_linear(linear->input_blob, linear->scale, linear->offset);
}

static int op_add(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_add_e2e(input);
    return run_add(chain->input_blob);
}

static int op_multiply(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_multiply_e2e(input);
    return run_multiply(chain->input_blob);
}

static int op_subtract(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_subtract_e2e(input);
    return run_subtract(chain->input_blob);
}

static int op_cast(const char *input, void *ctx) {
    struct cast_ctx *cast = (struct cast_ctx *)ctx;
    if (!cast->input_blob)
        return run_cast_e2e(input, cast->target);
    return run_cast(cast->input_blob, cast->target);
}

static int op_flip(const char *input, void *ctx) {
    struct flip_ctx *flip = (struct flip_ctx *)ctx;
    if (!flip->input_blob)
        return run_flip_e2e(input, flip->direction);
    return run_flip(flip->input_blob, flip->direction);
}

static int op_gamma(const char *input, void *ctx) {
    struct gamma_ctx *gamma = (struct gamma_ctx *)ctx;
    if (!gamma->input_blob)
        return run_gamma_e2e(input, gamma->exponent);
    return run_gamma(gamma->input_blob, gamma->exponent);
}

static int op_composite(const char *input, void *ctx) {
    struct composite_ctx *composite = (struct composite_ctx *)ctx;
    if (!composite->input_blob)
        return run_composite_e2e(input, composite->mode);
    return run_composite(composite->input_blob, composite->mode);
}

static int op_gauss_blur(const char *input, void *ctx) {
    struct gauss_blur_ctx *gauss_blur = (struct gauss_blur_ctx *)ctx;
    if (!gauss_blur->input_blob)
        return run_gauss_blur_e2e(input, gauss_blur->sigma);
    return run_gauss_blur(gauss_blur->input_blob, gauss_blur->sigma);
}

static int op_colourspace(const char *input, void *ctx) {
    struct colourspace_ctx *colourspace = (struct colourspace_ctx *)ctx;
    if (!colourspace->input_blob)
        return run_colourspace_e2e(input,
                                   colourspace->targets,
                                   colourspace->target_count);
    return run_colourspace(colourspace->input_blob,
                           colourspace->targets,
                           colourspace->target_count);
}

static int op_srgb_to_lab(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_srgb_to_lab_e2e(input);
    return run_srgb_to_lab(chain->input_blob);
}

static int op_invert(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_invert_e2e(input);
    VipsImage *in = new_input_image(invert->input_blob);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_invert(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    int sink_ret = vips_sink_discard(out);
    g_object_unref(out);
    return sink_ret;
}

static int op_abs(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_abs_e2e(input);
    return run_abs(invert->input_blob);
}

static int op_sign(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_sign_e2e(input);
    return run_sign(invert->input_blob);
}

static int op_round(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_round_e2e(input);
    return run_round(invert->input_blob);
}

static int op_floor(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_floor_e2e(input);
    return run_floor_op(invert->input_blob);
}

static int op_ceil(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_ceil_e2e(input);
    return run_ceil_op(invert->input_blob);
}

static int op_invert_invert(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_invert_invert_e2e(input);
    return run_invert_invert(invert->input_blob);
}

static int op_bandmean(const char *input, void *ctx) {
    struct bandmean_ctx *bandmean = (struct bandmean_ctx *)ctx;
    if (!bandmean->input_blob)
        return run_bandmean_e2e(input);
    return run_bandmean(bandmean->input_blob);
}

static int op_and(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_and_e2e(input);
    return run_and(invert->input_blob);
}

static int op_equal(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_equal_e2e(input);
    return run_equal(invert->input_blob);
}

static int op_histogram(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_histogram_e2e(input);
    return run_histogram(invert->input_blob);
}

static int op_recomb(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_recomb_e2e(input);
    return run_recomb(invert->input_blob);
}

static int op_grey(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_grey_e2e(input);
    return run_grey(invert->input_blob->width, invert->input_blob->height);
}

static int op_draw_line(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_draw_line_e2e(input);
    return run_draw_line(
        invert->input_blob->width,
        invert->input_blob->height,
        invert->input_blob->bands
    );
}

static int op_draw_rect(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_draw_rect_e2e(input);
    return run_draw_rect(
        invert->input_blob->width,
        invert->input_blob->height,
        invert->input_blob->bands
    );
}

static int op_draw_circle(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_draw_circle_e2e(input);
    return run_draw_circle(
        invert->input_blob->width,
        invert->input_blob->height,
        invert->input_blob->bands
    );
}

static int op_freqfilt(const char *input, void *ctx) {
    struct invert_ctx *invert = (struct invert_ctx *)ctx;
    if (!invert->input_blob)
        return run_freqfilt_e2e(input);
    return run_freqfilt(invert->input_blob);
}

static int op_thumbnail_sharpen(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_thumbnail_sharpen_e2e(input);
    return run_thumbnail_sharpen(chain->input_blob);
}

static int op_thumbnail_colourspace_cast(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_thumbnail_colourspace_cast_e2e(input);
    return run_thumbnail_colourspace_cast(chain->input_blob);
}

static int op_thumbnail_gauss_blur(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_thumbnail_gauss_blur_e2e(input);
    return run_thumbnail_gauss_blur(chain->input_blob);
}

static int op_thumbnail_linear(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_thumbnail_linear_e2e(input);
    return run_thumbnail_linear(chain->input_blob);
}

static int op_resize_colourspace(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_resize_colourspace_e2e(input);
    return run_resize_colourspace(chain->input_blob);
}

static int op_embed(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_embed_e2e(input);
    return run_embed(chain->input_blob);
}

static int op_extract_area(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_extract_area_e2e(input);
    return run_extract_area(chain->input_blob);
}

static int op_embed_extract(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_embed_extract_e2e(input);
    return run_embed_extract(chain->input_blob);
}

static int op_three_op_chain(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_three_op_chain_e2e(input);
    return run_three_op_chain(chain->input_blob);
}

static int op_mapim(const char *input, void *ctx) {
    struct mapim_ctx *mapim = (struct mapim_ctx *)ctx;
    if (!mapim->input_blob)
        return run_mapim_e2e(input, mapim->index_buf, mapim->width, mapim->height);
    return run_mapim(mapim->input_blob, mapim->index_buf);
}

static int op_sharpen(const char *input, void *ctx) {
    struct sharpen_ctx *sharpen = (struct sharpen_ctx *)ctx;
    if (!sharpen->input_blob)
        return run_sharpen_e2e(input, sharpen->sigma, sharpen->strength);
    return run_sharpen(sharpen->input_blob, sharpen->sigma, sharpen->strength);
}

static int op_unsharp_mask(const char *input, void *ctx) {
    struct sharpen_ctx *sharpen = (struct sharpen_ctx *)ctx;
    if (!sharpen->input_blob)
        return run_unsharp_mask_e2e(input, sharpen->sigma, sharpen->strength);
    return run_unsharp_mask(sharpen->input_blob, sharpen->sigma, sharpen->strength);
}

static int op_conv(const char *input, void *ctx) {
    struct conv_ctx *conv = (struct conv_ctx *)ctx;
    if (!conv->input_blob)
        return run_conv_e2e(input, conv->coeff, conv->kernel_size, conv->scale, conv->offset);
    return run_conv(conv->input_blob, conv->coeff, conv->kernel_size, conv->scale, conv->offset);
}

static int op_sobel(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_sobel_e2e(input);
    return run_sobel(chain->input_blob);
}

static int op_prewitt(const char *input, void *ctx) {
    struct chain_ctx *chain = (struct chain_ctx *)ctx;
    if (!chain->input_blob)
        return run_prewitt_e2e(input);
    return run_prewitt(chain->input_blob);
}

static int op_median_blur(const char *input, void *ctx) {
    struct morphology_ctx *morphology = (struct morphology_ctx *)ctx;
    if (!morphology->input_blob)
        return run_median_blur_e2e(input, morphology->kernel_size);
    return run_median_blur(morphology->input_blob, morphology->kernel_size);
}

static int op_dilate(const char *input, void *ctx) {
    struct morphology_ctx *morphology = (struct morphology_ctx *)ctx;
    if (!morphology->input_blob)
        return run_morphology_e2e(input,
                                  morphology->kernel_size,
                                  VIPS_OPERATION_MORPHOLOGY_DILATE);
    return run_morphology(morphology->input_blob,
                          morphology->kernel_size,
                          VIPS_OPERATION_MORPHOLOGY_DILATE);
}

static int op_erode(const char *input, void *ctx) {
    struct morphology_ctx *morphology = (struct morphology_ctx *)ctx;
    if (!morphology->input_blob)
        return run_morphology_e2e(input,
                                  morphology->kernel_size,
                                  VIPS_OPERATION_MORPHOLOGY_ERODE);
    return run_morphology(morphology->input_blob,
                          morphology->kernel_size,
                          VIPS_OPERATION_MORPHOLOGY_ERODE);
}

static int op_open(const char *input, void *ctx) {
    struct morphology_ctx *morphology = (struct morphology_ctx *)ctx;
    if (!morphology->input_blob)
        return run_open_e2e(input, morphology->kernel_size);
    return run_open(morphology->input_blob, morphology->kernel_size);
}

static int op_close(const char *input, void *ctx) {
    struct morphology_ctx *morphology = (struct morphology_ctx *)ctx;
    if (!morphology->input_blob)
        return run_close_e2e(input, morphology->kernel_size);
    return run_close(morphology->input_blob, morphology->kernel_size);
}

static int op_save_avif(const char *input, void *ctx) {
    struct save_avif_ctx *save_avif = (struct save_avif_ctx *)ctx;
    if (!save_avif->input_blob)
        return run_save_avif_e2e(input);
    return run_save_avif(save_avif->input_blob);
}

static int op_save_exr(const char *input, void *ctx) {
    struct save_exr_ctx *save_exr = (struct save_exr_ctx *)ctx;
    if (!save_exr->input_blob)
        return run_save_exr_e2e(input);
    return run_save_exr(save_exr->input_blob);
}

static int op_save_gif(const char *input, void *ctx) {
    struct save_gif_ctx *save_gif = (struct save_gif_ctx *)ctx;
    if (!save_gif->input_blob)
        return run_save_gif_e2e(input);
    return run_save_gif(save_gif->input_blob);
}

static int op_save_heif(const char *input, void *ctx) {
    struct save_heif_ctx *save_heif = (struct save_heif_ctx *)ctx;
    if (!save_heif->input_blob)
        return run_save_heif_e2e(input);
    return run_save_heif(save_heif->input_blob);
}

static int op_save_jpeg(const char *input, void *ctx) {
    struct save_jpeg_ctx *save_jpeg = (struct save_jpeg_ctx *)ctx;
    if (!save_jpeg->input_blob)
        return run_save_jpeg_e2e(input);
    return run_save_jpeg(save_jpeg->input_blob);
}

static int op_save_jp2k(const char *input, void *ctx) {
    struct save_jp2k_ctx *save_jp2k = (struct save_jp2k_ctx *)ctx;
    if (!save_jp2k->input_blob)
        return run_save_jp2k_e2e(input);
    return run_save_jp2k(save_jp2k->input_blob);
}

static int op_save_tiff(const char *input, void *ctx) {
    struct save_tiff_ctx *save_tiff = (struct save_tiff_ctx *)ctx;
    if (!save_tiff->input_blob)
        return run_save_tiff_e2e(input, save_tiff->compression);
    return run_save_tiff(save_tiff->input_blob, save_tiff->compression);
}

static int op_workflow(const char *input, void *ctx) {
    struct workflow_ctx *wf = (struct workflow_ctx *)ctx;
    if (!wf->input_blob)
        return run_workflow_e2e(input, wf->width, wf->target_format);
    return run_workflow(wf->input_blob, wf->width, wf->target_format);
}

static int op_load(const char *input, void *ctx) {
    (void)ctx;
    return run_load(input);
}

static int op_load_jpeg(const char *input, void *ctx) {
    (void)ctx;
    return run_load_jpeg(input);
}

int main(int argc, char **argv) {
    if (VIPS_INIT(argv[0])) {
        vips_error_exit(NULL);
    }

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char *input = argv[1];
    const char *op_name = canonicalize_op_name(argv[2]);
    int iterations = DEFAULT_ITERATIONS;
    int threads = 0;
    int libvips_cache_enabled = 0;
    int e2e = 0;
    int quiet = 0;

    /* Parse flags */
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[i + 1]);
            if (iterations <= 0 || iterations > MAX_ITERATIONS)
                iterations = DEFAULT_ITERATIONS;
            i++;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = atoi(argv[i + 1]);
            if (threads <= 0) {
                fprintf(stderr, "--threads requires a non-zero integer worker count\n");
                vips_shutdown();
                return 1;
            }
            i++;
        } else if (strcmp(argv[i], "--libvips-cache") == 0) {
            libvips_cache_enabled = 1;
        } else if (strcmp(argv[i], "--e2e") == 0) {
            e2e = 1;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            quiet = 1;
        }
    }

    if (threads == 0) {
        fprintf(stderr, "--threads is required\n");
        vips_shutdown();
        return 1;
    }

    vips_concurrency_set(threads);

    /* Disable libvips operation cache unless explicitly requested */
    if (!libvips_cache_enabled) {
        vips_cache_set_max(0);
    }

    if (strcmp(op_name, "load") == 0 || strcmp(op_name, "load-jpeg") == 0) {
        op_fn fn = strcmp(op_name, "load-jpeg") == 0 ? op_load_jpeg : op_load;

        /* Warmup (3 iterations, discard) */
        for (int i = 0; i < 3; i++) {
            fn(input, NULL);
        }

        struct rusage ru_before, ru_after;
        getrusage(RUSAGE_SELF, &ru_before);

        long long *wall_ns = malloc(sizeof(long long) * iterations);
        struct timespec t0, t1;

        for (int i = 0; i < iterations; i++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            int ret = fn(input, NULL);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            if (ret != 0) {
                fprintf(stderr, "Operation failed at iteration %d\n", i);
                free(wall_ns);
                vips_shutdown();
                return 1;
            }
            wall_ns[i] = timespec_diff_ns(&t0, &t1);
        }

        getrusage(RUSAGE_SELF, &ru_after);

        if (!quiet) {
            printf("{\n");
            printf("  \"backend\": \"libvips\",\n");
            printf("  \"version\": \"%s\",\n", vips_version_string());
            printf("  \"input\": \"%s\",\n", input);
            printf("  \"operation\": \"%s\",\n", op_name);
            printf("  \"iterations\": %d,\n", iterations);
            printf("  \"wall_ns\": [");
            for (int i = 0; i < iterations; i++) {
                printf("%lld%s", wall_ns[i], i < iterations - 1 ? "," : "");
            }
            printf("],\n");
            printf("  \"peak_rss_kb\": %ld,\n", ru_after.ru_maxrss / 1024);
            printf("  \"minor_faults\": %ld,\n",
                   ru_after.ru_minflt - ru_before.ru_minflt);
            printf("  \"major_faults\": %ld,\n",
                   ru_after.ru_majflt - ru_before.ru_majflt);
            printf("  \"vol_ctx_switches\": %ld,\n",
                   ru_after.ru_nvcsw - ru_before.ru_nvcsw);
            printf("  \"invol_ctx_switches\": %ld\n",
                   ru_after.ru_nivcsw - ru_before.ru_nivcsw);
            printf("}\n");
        }

        free(wall_ns);
        vips_shutdown();
        return 0;
    }

    /* Select operation */
    op_fn fn = NULL;
    void *ctx = NULL;
    static const double conv_sharpen3_coeff[9] = {
        0.0, -1.0, 0.0,
        -1.0, 5.0, -1.0,
        0.0, -1.0, 0.0
    };
    static const double conv_box3_coeff[9] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0
    };
    static const double conv_sobel3_coeff[9] = {
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0
    };
    static const double conv_laplacian3_coeff[9] = {
        0.0, -1.0, 0.0,
        -1.0, 4.0, -1.0,
        0.0, -1.0, 0.0
    };
    struct thumbnail_ctx thumb_ctx;
    struct resize_ctx resize_ctx;
    struct zoom_ctx zoom_ctx;
    struct affine_ctx affine_ctx;
    struct similarity_ctx similarity_ctx;
    struct composite_shrink_ctx shrink_ctx;
    struct shrink_ctx shrinkh_ctx;
    struct shrink_ctx shrinkv_ctx;
    struct linear_ctx linear_ctx;
    struct cast_ctx cast_ctx;
    struct composite_ctx composite_ctx;
    struct flip_ctx flip_ctx;
    struct gauss_blur_ctx gauss_blur_ctx;
    struct gamma_ctx gamma_ctx;
    struct colourspace_ctx colourspace_ctx = { 0 };
    struct invert_ctx invert_ctx = { 0 };
    struct bandmean_ctx bandmean_ctx = { 0 };
    struct chain_ctx chain_ctx = { 0 };
    struct sharpen_ctx sharpen_ctx = { 0 };
    struct conv_ctx conv_ctx = { 0 };
    struct morphology_ctx morphology_ctx = { 0 };
    struct save_avif_ctx save_avif_ctx = { 0 };
    struct save_exr_ctx save_exr_ctx = { 0 };
    struct save_gif_ctx save_gif_ctx = { 0 };
    struct save_heif_ctx save_heif_ctx = { 0 };
    struct save_jpeg_ctx save_jpeg_ctx = { 0 };
    struct save_jp2k_ctx save_jp2k_ctx = { 0 };
    struct save_tiff_ctx save_tiff_ctx = { 0 };
    struct mapim_ctx mapim_ctx = { 0 };
    struct input_blob input_blob = { 0 };

    /*
     * In --e2e mode each op function decodes from disk on every call, so we
     * skip pre-loading the image to memory. The input_blob fields stay zero
     * (NULL raw_buf) and each ctx's input_blob pointer stays NULL, which the
     * op functions check to select the e2e code path.
     */
    if (!e2e) {
        VipsImage *src_img = vips_image_new_from_file(input, NULL);
        if (!src_img) {
            fprintf(stderr, "Failed to load input image: %s\n", input);
            vips_shutdown();
            return 1;
        }
        input_blob.raw_buf = vips_image_write_to_memory(src_img, &input_blob.raw_len);
        input_blob.width = vips_image_get_width(src_img);
        input_blob.height = vips_image_get_height(src_img);
        input_blob.bands = vips_image_get_bands(src_img);
        input_blob.format = vips_image_get_format(src_img);
        input_blob.interpretation = src_img->Type;
        g_object_unref(src_img);
        if (!input_blob.raw_buf || input_blob.raw_len == 0) {
            fprintf(stderr, "Failed to preload input image to memory: %s\n", input);
            if (input_blob.raw_buf) {
                g_free(input_blob.raw_buf);
            }
            vips_shutdown();
            return 1;
        }
    }

    if (strcmp(op_name, "mapim") == 0) {
        mapim_ctx.dx = (argc > 3 && argv[3][0] != '-') ? atof(argv[3]) : DEFAULT_MAPIM_DX;
        mapim_ctx.dy = (argc > 4 && argv[4][0] != '-') ? atof(argv[4]) : DEFAULT_MAPIM_DY;
        if (e2e) {
            VipsImage *dims = vips_image_new_from_file(input, NULL);
            if (!dims) {
                vips_shutdown();
                return 1;
            }
            mapim_ctx.width = vips_image_get_width(dims);
            mapim_ctx.height = vips_image_get_height(dims);
            g_object_unref(dims);
        } else {
            mapim_ctx.width = input_blob.width;
            mapim_ctx.height = input_blob.height;
        }
        if (prepare_mapim_index(&mapim_ctx) != 0) {
            fprintf(stderr, "Failed to prepare mapim index image\n");
            if (input_blob.raw_buf) g_free(input_blob.raw_buf);
            vips_shutdown();
            return 1;
        }
    }

    if (strcmp(op_name, "load") == 0) {
        fn = op_load;
        ctx = NULL;
    } else if (strcmp(op_name, "save-avif") == 0) {
        save_avif_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_save_avif;
        ctx = &save_avif_ctx;
    } else if (strcmp(op_name, "save-exr") == 0) {
        save_exr_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_save_exr;
        ctx = &save_exr_ctx;
    } else if (strcmp(op_name, "save-gif") == 0) {
        save_gif_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_save_gif;
        ctx = &save_gif_ctx;
    } else if (strcmp(op_name, "save-heif") == 0) {
        save_heif_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_save_heif;
        ctx = &save_heif_ctx;
    } else if (strcmp(op_name, "save-jpeg") == 0) {
        save_jpeg_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_save_jpeg;
        ctx = &save_jpeg_ctx;
    } else if (strcmp(op_name, "save-jp2k") == 0) {
        save_jp2k_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_save_jp2k;
        ctx = &save_jp2k_ctx;
    } else if (strcmp(op_name, "save-tiff") == 0) {
        const char *compression_arg = (argc > 3 && argv[3][0] != '-') ? argv[3] : NULL;
        save_tiff_ctx.input_blob = e2e ? NULL : &input_blob;
        save_tiff_ctx.compression = parse_tiff_compression_arg(compression_arg);
        fn = op_save_tiff;
        ctx = &save_tiff_ctx;
    } else if (strcmp(op_name, "thumbnail") == 0) {
        thumb_ctx.width = (argc > 3 && argv[3][0] != '-') ? atoi(argv[3]) : 800;
        thumb_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_thumbnail;
        ctx = &thumb_ctx;
    } else if (strcmp(op_name, "resize") == 0) {
        resize_ctx.scale = (argc > 3 && argv[3][0] != '-') ? atof(argv[3]) : 0.5;
        resize_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_resize;
        ctx = &resize_ctx;
    } else if (strcmp(op_name, "zoom") == 0) {
        zoom_ctx.xfac = (argc > 3 && argv[3][0] != '-') ? atoi(argv[3]) : 2;
        zoom_ctx.yfac = (argc > 4 && argv[4][0] != '-') ? atoi(argv[4]) : zoom_ctx.xfac;
        zoom_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_zoom;
        ctx = &zoom_ctx;
    } else if (strcmp(op_name, "affine") == 0) {
        affine_ctx.a = (argc > 3 && argv[3][0] != '-') ? atof(argv[3]) : DEFAULT_AFFINE_A;
        affine_ctx.b = (argc > 4 && argv[4][0] != '-') ? atof(argv[4]) : DEFAULT_AFFINE_B;
        affine_ctx.c = (argc > 5 && argv[5][0] != '-') ? atof(argv[5]) : DEFAULT_AFFINE_C;
        affine_ctx.d = (argc > 6 && argv[6][0] != '-') ? atof(argv[6]) : DEFAULT_AFFINE_D;
        affine_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_affine;
        ctx = &affine_ctx;
    } else if (strcmp(op_name, "similarity") == 0) {
        similarity_ctx.scale = (argc > 3 && argv[3][0] != '-') ? atof(argv[3]) : DEFAULT_SIMILARITY_SCALE;
        similarity_ctx.angle = (argc > 4 && argv[4][0] != '-') ? atof(argv[4]) : DEFAULT_SIMILARITY_ANGLE;
        similarity_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_similarity;
        ctx = &similarity_ctx;
    } else if (strcmp(op_name, "shrink") == 0) {
        shrink_ctx.hfactor = (argc > 3 && argv[3][0] != '-') ? atoi(argv[3]) : 2;
        shrink_ctx.vfactor = (argc > 4 && argv[4][0] != '-') ? atoi(argv[4]) : shrink_ctx.hfactor;
        shrink_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_shrink;
        ctx = &shrink_ctx;
    } else if (strcmp(op_name, "shrinkh") == 0) {
        shrinkh_ctx.factor = (argc > 3 && argv[3][0] != '-') ? atoi(argv[3]) : 2;
        shrinkh_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_shrinkh;
        ctx = &shrinkh_ctx;
    } else if (strcmp(op_name, "shrinkv") == 0) {
        shrinkv_ctx.factor = (argc > 3 && argv[3][0] != '-') ? atoi(argv[3]) : 2;
        shrinkv_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_shrinkv;
        ctx = &shrinkv_ctx;
    } else if (strcmp(op_name, "linear") == 0) {
        linear_ctx.scale = (argc > 3 && argv[3][0] != '-') ? atof(argv[3]) : 2.0;
        linear_ctx.offset = (argc > 4 && argv[4][0] != '-') ? atof(argv[4]) : 5.0;
        linear_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_linear;
        ctx = &linear_ctx;
    } else if (strcmp(op_name, "abs") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_abs;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "sign") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_sign;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "round") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_round;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "floor") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_floor;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "ceil") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_ceil;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "add") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_add;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "multiply") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_multiply;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "subtract") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_subtract;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "and") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_and;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "equal") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_equal;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "cast") == 0) {
        const char *target_arg = (argc > 3 && argv[3][0] != '-') ? argv[3] : NULL;
        cast_ctx.input_blob = e2e ? NULL : &input_blob;
        cast_ctx.target = parse_cast_target_arg(e2e ? VIPS_FORMAT_UCHAR : input_blob.format, target_arg);
        fn = op_cast;
        ctx = &cast_ctx;
    } else if (strcmp(op_name, "flip") == 0) {
        const char *direction_arg = (argc > 3 && argv[3][0] != '-') ? argv[3] : NULL;
        flip_ctx.input_blob = e2e ? NULL : &input_blob;
        flip_ctx.direction = parse_flip_direction_arg(direction_arg);
        fn = op_flip;
        ctx = &flip_ctx;
    } else if (strcmp(op_name, "gamma") == 0) {
        gamma_ctx.input_blob = e2e ? NULL : &input_blob;
        gamma_ctx.exponent = (argc > 3 && argv[3][0] != '-') ? atof(argv[3]) : DEFAULT_GAMMA_EXPONENT;
        fn = op_gamma;
        ctx = &gamma_ctx;
    } else if (strcmp(op_name, "composite") == 0) {
        const char *mode_arg = (argc > 3 && argv[3][0] != '-') ? argv[3] : NULL;
        composite_ctx.mode = parse_composite_mode_arg(mode_arg);
        composite_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_composite;
        ctx = &composite_ctx;
    } else if (strcmp(op_name, "gauss_blur") == 0) {
        gauss_blur_ctx.sigma = (argc > 3 && argv[3][0] != '-') ? atof(argv[3]) : 1.5;
        gauss_blur_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_gauss_blur;
        ctx = &gauss_blur_ctx;
    } else if (strcmp(op_name, "convolve") == 0) {
        conv_ctx.input_blob = e2e ? NULL : &input_blob;
        conv_ctx.coeff = conv_box3_coeff;
        conv_ctx.kernel_size = 3;
        conv_ctx.scale = 9.0;
        conv_ctx.offset = 0.0;
        fn = op_conv;
        ctx = &conv_ctx;
    } else if (strcmp(op_name, "sobel") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_sobel;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "prewitt") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_prewitt;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "laplacian") == 0) {
        conv_ctx.input_blob = e2e ? NULL : &input_blob;
        conv_ctx.coeff = conv_laplacian3_coeff;
        conv_ctx.kernel_size = 3;
        conv_ctx.scale = 1.0;
        conv_ctx.offset = 0.0;
        fn = op_conv;
        ctx = &conv_ctx;
    } else if (strcmp(op_name, "median_blur") == 0) {
        morphology_ctx.input_blob = e2e ? NULL : &input_blob;
        morphology_ctx.kernel_size = (argc > 3 && argv[3][0] != '-') ? atoi(argv[3]) : 3;
        fn = op_median_blur;
        ctx = &morphology_ctx;
    } else if (strcmp(op_name, "unsharp_mask") == 0) {
        sharpen_ctx.input_blob = e2e ? NULL : &input_blob;
        sharpen_ctx.sigma = (argc > 3 && argv[3][0] != '-') ? atof(argv[3]) : 0.5;
        sharpen_ctx.strength = (argc > 4 && argv[4][0] != '-') ? atof(argv[4]) : 3.0;
        fn = op_unsharp_mask;
        ctx = &sharpen_ctx;
    } else if (strcmp(op_name, "colourspace") == 0) {
        colourspace_ctx.input_blob = e2e ? NULL : &input_blob;
        colourspace_ctx.target_count = 0;
        for (int i = 3; i < argc && argv[i][0] != '-'; i++) {
            if (colourspace_ctx.target_count >= MAX_COLOURSPACE_STEPS) {
                fprintf(stderr, "colourspace accepts at most %d destination steps\n",
                        MAX_COLOURSPACE_STEPS);
                if (input_blob.raw_buf) g_free(input_blob.raw_buf);
                vips_shutdown();
                return 1;
            }
            colourspace_ctx.targets[colourspace_ctx.target_count++] =
                parse_colourspace_arg(argv[i]);
        }
        fn = op_colourspace;
        ctx = &colourspace_ctx;
    } else if (strcmp(op_name, "srgb_to_lab") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_srgb_to_lab;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "invert") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_invert;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "invert_invert") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_invert_invert;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "bandmean") == 0) {
        bandmean_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_bandmean;
        ctx = &bandmean_ctx;
    } else if (strcmp(op_name, "histogram") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_histogram;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "recomb") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_recomb;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "grey") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_grey;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "draw_line") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_draw_line;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "draw_rect") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_draw_rect;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "draw_circle") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_draw_circle;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "freqfilt") == 0) {
        invert_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_freqfilt;
        ctx = &invert_ctx;
    } else if (strcmp(op_name, "thumbnail_sharpen") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_thumbnail_sharpen;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "thumbnail_colourspace_cast") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_thumbnail_colourspace_cast;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "thumbnail_gauss_blur") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_thumbnail_gauss_blur;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "thumbnail_linear") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_thumbnail_linear;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "resize_colourspace") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_resize_colourspace;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "embed") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_embed;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "extract-area") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_extract_area;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "embed_extract") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_embed_extract;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "three_op_chain") == 0) {
        chain_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_three_op_chain;
        ctx = &chain_ctx;
    } else if (strcmp(op_name, "sharpen") == 0) {
        sharpen_ctx.input_blob = e2e ? NULL : &input_blob;
        sharpen_ctx.sigma = (argc > 3 && argv[3][0] != '-') ? atof(argv[3]) : 0.5;
        sharpen_ctx.strength = (argc > 4 && argv[4][0] != '-') ? atof(argv[4]) : 3.0;
        fn = op_sharpen;
        ctx = &sharpen_ctx;
    } else if (strcmp(op_name, "conv_sharpen3") == 0) {
        conv_ctx.input_blob = e2e ? NULL : &input_blob;
        conv_ctx.coeff = conv_sharpen3_coeff;
        conv_ctx.kernel_size = 3;
        conv_ctx.scale = 1.0;
        conv_ctx.offset = 0.0;
        fn = op_conv;
        ctx = &conv_ctx;
    } else if (strcmp(op_name, "conv_sobel3") == 0) {
        conv_ctx.input_blob = e2e ? NULL : &input_blob;
        conv_ctx.coeff = conv_sobel3_coeff;
        conv_ctx.kernel_size = 3;
        conv_ctx.scale = 1.0;
        conv_ctx.offset = 0.0;
        fn = op_conv;
        ctx = &conv_ctx;
    } else if (strcmp(op_name, "dilate") == 0) {
        morphology_ctx.input_blob = e2e ? NULL : &input_blob;
        morphology_ctx.kernel_size = (argc > 3 && argv[3][0] != '-') ? atoi(argv[3]) : 3;
        fn = op_dilate;
        ctx = &morphology_ctx;
    } else if (strcmp(op_name, "erode") == 0) {
        morphology_ctx.input_blob = e2e ? NULL : &input_blob;
        morphology_ctx.kernel_size = (argc > 3 && argv[3][0] != '-') ? atoi(argv[3]) : 3;
        fn = op_erode;
        ctx = &morphology_ctx;
    } else if (strcmp(op_name, "open") == 0) {
        morphology_ctx.input_blob = e2e ? NULL : &input_blob;
        morphology_ctx.kernel_size = (argc > 3 && argv[3][0] != '-') ? atoi(argv[3]) : 3;
        fn = op_open;
        ctx = &morphology_ctx;
    } else if (strcmp(op_name, "close") == 0) {
        morphology_ctx.input_blob = e2e ? NULL : &input_blob;
        morphology_ctx.kernel_size = (argc > 3 && argv[3][0] != '-') ? atoi(argv[3]) : 3;
        fn = op_close;
        ctx = &morphology_ctx;
    } else if (strcmp(op_name, "mapim") == 0) {
        mapim_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_mapim;
        ctx = &mapim_ctx;
    } else if (strcmp(op_name, "workflow") == 0) {
        static struct workflow_ctx wf_ctx;
        const char *fmt_arg = (argc > 3 && argv[3][0] != '-') ? argv[3] : "webp";
        /* Map short names to libvips suffix strings */
        if (strcmp(fmt_arg, "jpg") == 0 || strcmp(fmt_arg, "jpeg") == 0)
            wf_ctx.target_format = ".jpg";
        else if (strcmp(fmt_arg, "webp") == 0)
            wf_ctx.target_format = ".webp";
        else if (strcmp(fmt_arg, "avif") == 0)
            wf_ctx.target_format = ".avif";
        else if (strcmp(fmt_arg, "png") == 0)
            wf_ctx.target_format = ".png";
        else if (strcmp(fmt_arg, "tif") == 0 || strcmp(fmt_arg, "tiff") == 0)
            wf_ctx.target_format = ".tif";
        else {
            fprintf(stderr, "workflow: unsupported target format '%s' (use jpg, webp, avif, png, tif)\n", fmt_arg);
            if (input_blob.raw_buf) g_free(input_blob.raw_buf);
            vips_shutdown();
            return 1;
        }
        wf_ctx.width = (argc > 4 && argv[4][0] != '-') ? atoi(argv[4]) : 400;
        wf_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_workflow;
        ctx = &wf_ctx;
    } else if (strcmp(op_name, "perceptual_enhance") == 0) {
        static struct workflow_ctx pe_ctx;
        const char *fmt_arg = (argc > 3 && argv[3][0] != '-') ? argv[3] : "webp";
        if (strcmp(fmt_arg, "jpg") == 0 || strcmp(fmt_arg, "jpeg") == 0)
            pe_ctx.target_format = ".jpg";
        else if (strcmp(fmt_arg, "webp") == 0)
            pe_ctx.target_format = ".webp";
        else if (strcmp(fmt_arg, "avif") == 0)
            pe_ctx.target_format = ".avif";
        else if (strcmp(fmt_arg, "png") == 0)
            pe_ctx.target_format = ".png";
        else if (strcmp(fmt_arg, "tif") == 0 || strcmp(fmt_arg, "tiff") == 0)
            pe_ctx.target_format = ".tif";
        else
            pe_ctx.target_format = ".webp";
        pe_ctx.width = (argc > 4 && argv[4][0] != '-') ? atoi(argv[4]) : 800;
        pe_ctx.input_blob = e2e ? NULL : &input_blob;
        fn = op_perceptual_enhance;
        ctx = &pe_ctx;
    } else {
        fprintf(stderr, "Unknown operation: %s\n", op_name);
        if (input_blob.raw_buf) g_free(input_blob.raw_buf);
        vips_shutdown();
        return 1;
    }

    /* Warmup (3 iterations, discard) */
    for (int i = 0; i < 3; i++) {
        fn(input, ctx);
    }

    /* Collect getrusage before */
    struct rusage ru_before, ru_after;
    getrusage(RUSAGE_SELF, &ru_before);

    /* Benchmark iterations */
    long long *wall_ns = malloc(sizeof(long long) * iterations);
    struct timespec t0, t1;

    for (int i = 0; i < iterations; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int ret = fn(input, ctx);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (ret != 0) {
            fprintf(stderr, "Operation failed at iteration %d\n", i);
            free(wall_ns);
            if (input_blob.raw_buf) g_free(input_blob.raw_buf);
            if (mapim_ctx.index_buf) g_free(mapim_ctx.index_buf);
            vips_shutdown();
            return 1;
        }
        wall_ns[i] = timespec_diff_ns(&t0, &t1);
    }

    /* Collect getrusage after */
    getrusage(RUSAGE_SELF, &ru_after);

    /* Output JSON */
    if (!quiet) {
        printf("{\n");
        printf("  \"backend\": \"libvips\",\n");
        printf("  \"version\": \"%s\",\n", vips_version_string());
        printf("  \"input\": \"%s\",\n", input);
        printf("  \"operation\": \"%s\",\n", op_name);
        printf("  \"iterations\": %d,\n", iterations);
        printf("  \"wall_ns\": [");
        for (int i = 0; i < iterations; i++) {
            printf("%lld%s", wall_ns[i], i < iterations - 1 ? "," : "");
        }
        printf("],\n");
        printf("  \"peak_rss_kb\": %ld,\n", ru_after.ru_maxrss / 1024);
        printf("  \"minor_faults\": %ld,\n",
               ru_after.ru_minflt - ru_before.ru_minflt);
        printf("  \"major_faults\": %ld,\n",
               ru_after.ru_majflt - ru_before.ru_majflt);
        printf("  \"vol_ctx_switches\": %ld,\n",
               ru_after.ru_nvcsw - ru_before.ru_nvcsw);
        printf("  \"invol_ctx_switches\": %ld\n",
               ru_after.ru_nivcsw - ru_before.ru_nivcsw);
        printf("}\n");
    }

    free(wall_ns);
    if (input_blob.raw_buf) g_free(input_blob.raw_buf);
    if (mapim_ctx.index_buf) g_free(mapim_ctx.index_buf);
    vips_shutdown();
    return 0;
}
