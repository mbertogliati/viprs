/*
 * libvips web-service benchmark runner.
 *
 * Simulates web-service image processing: read bytes from buffer (as if from
 * HTTP body), decode, transform, encode to buffer, discard. No disk I/O in
 * the hot loop.
 *
 * Usage:
 *   ./libvips-web-runner <input_path> <scenario> [args...] --iterations N
 *
 * Scenarios:
 *   thumbnail-bytes <width>                          - buffer → thumbnail → encode WebP → buffer
 *   pipeline-bytes <width> <quality>                - buffer → thumbnail → sharpen → linear → JPEG
 *   concurrent <width> <concurrency> <iters/thread> - parallel thumbnail_buffer requests
 *   large-upload <width>                            - decode large buffer → thumbnail → encode WebP
 *   workflow <format> [width]                       - buffer → thumbnail → sharpen → encode format
 *
 * Output: JSON with wall_ns[] array and metrics.
 */

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vips/vips.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

#define MAX_ITERATIONS 10000
#define DEFAULT_ITERATIONS 30

static long long timespec_diff_ns(const struct timespec *start, const struct timespec *end) {
    return (long long)(end->tv_sec - start->tv_sec) * 1000000000LL
         + (long long)(end->tv_nsec - start->tv_nsec);
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <input> <scenario> [args...] --iterations N\n"
        "\n"
        "Scenarios:\n"
        "  thumbnail-bytes [width]                  decode buffer → thumbnail → WebP buffer\n"
        "  pipeline-bytes [width] [quality]         decode → thumbnail → sharpen → linear → JPEG\n"
        "  concurrent [width] [concurrency] [iters] parallel thumbnail_buffer requests\n"
        "  large-upload [width]                     decode large buffer → thumbnail → WebP buffer\n"
        "  workflow <format> [width]                decode → thumbnail → sharpen → encode\n"
        "\n"
        "Formats: webp, jpg, png\n",
        prog);
}

static long current_resident_kb(void) {
#if defined(__linux__)
    FILE *file = fopen("/proc/self/statm", "r");
    if (!file) {
        return 0;
    }

    unsigned long total_pages = 0;
    unsigned long resident_pages = 0;
    if (fscanf(file, "%lu %lu", &total_pages, &resident_pages) != 2) {
        fclose(file);
        return 0;
    }
    fclose(file);

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return 0;
    }
    return (long)((resident_pages * (unsigned long)page_size) / 1024UL);
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t status =
        task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count);
    if (status != KERN_SUCCESS) {
        return 0;
    }
    return (long)(info.resident_size / 1024UL);
#else
    return 0;
#endif
}

static int run_thumbnail_bytes(const void *file_buf, size_t file_len, int width) {
    VipsImage *im = vips_image_new_from_buffer(file_buf, file_len, "", NULL);
    if (!im) return -1;

    VipsImage *thumb = NULL;
    int ret = vips_thumbnail_image(im, &thumb, width, NULL);
    g_object_unref(im);
    if (ret != 0 || !thumb) return -1;

    void *out_buf = NULL;
    size_t out_len = 0;
    ret = vips_webpsave_buffer(thumb, &out_buf, &out_len, NULL);
    g_object_unref(thumb);
    if (ret != 0 || !out_buf) return -1;

    g_free(out_buf);
    return 0;
}

static int run_pipeline_bytes(const void *file_buf, size_t file_len, int width, int quality) {
    VipsImage *im = vips_image_new_from_buffer(file_buf, file_len, "", NULL);
    if (!im) return -1;

    VipsImage *thumb = NULL;
    int ret = vips_thumbnail_image(im, &thumb, width, NULL);
    g_object_unref(im);
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

    VipsImage *lin = NULL;
    double a[] = {1.1, 1.1, 1.1};
    double b[] = {5.0, 5.0, 5.0};
    ret = vips_linear(sharp, &lin, a, b, sharp->Bands, NULL);
    g_object_unref(sharp);
    if (ret != 0 || !lin) return -1;

    void *out_buf = NULL;
    size_t out_len = 0;
    ret = vips_jpegsave_buffer(lin, &out_buf, &out_len, "Q", quality, NULL);
    g_object_unref(lin);
    if (ret != 0 || !out_buf) return -1;

    g_free(out_buf);
    return 0;
}

static int run_workflow_web(const void *file_buf, size_t file_len, int width, const char *format) {
    VipsImage *im = vips_image_new_from_buffer(file_buf, file_len, "", NULL);
    if (!im) return -1;

    VipsImage *thumb = NULL;
    int ret = vips_thumbnail_image(im, &thumb, width, NULL);
    g_object_unref(im);
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

    void *out_buf = NULL;
    size_t out_len = 0;
    if (strcmp(format, "webp") == 0) {
        ret = vips_webpsave_buffer(sharp, &out_buf, &out_len, NULL);
    } else if (strcmp(format, "jpg") == 0 || strcmp(format, "jpeg") == 0) {
        ret = vips_jpegsave_buffer(sharp, &out_buf, &out_len, "Q", 85, NULL);
    } else if (strcmp(format, "png") == 0) {
        ret = vips_pngsave_buffer(sharp, &out_buf, &out_len, NULL);
    } else {
        fprintf(stderr, "Unsupported format: %s\n", format);
        g_object_unref(sharp);
        return -1;
    }

    g_object_unref(sharp);
    if (ret != 0 || !out_buf) return -1;

    g_free(out_buf);
    return 0;
}

typedef struct {
    const void *file_buf;
    size_t file_len;
    int width;
    int iterations;
    long long *latencies;
    int failed;
} concurrent_worker_args_t;

static void *run_concurrent_worker(void *arg) {
    concurrent_worker_args_t *worker = (concurrent_worker_args_t *)arg;
    for (int i = 0; i < worker->iterations; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (run_thumbnail_bytes(worker->file_buf, worker->file_len, worker->width) != 0) {
            worker->failed = 1;
            return NULL;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        worker->latencies[i] = timespec_diff_ns(&t0, &t1);
    }
    return NULL;
}

static int run_concurrent_scenario(
    const void *file_buf,
    size_t file_len,
    int width,
    int concurrency,
    int iterations_per_thread,
    long long *latencies,
    long long *wall_total_ns
) {
    pthread_t *threads = calloc((size_t)concurrency, sizeof(pthread_t));
    concurrent_worker_args_t *workers =
        calloc((size_t)concurrency, sizeof(concurrent_worker_args_t));
    if (!threads || !workers) {
        free(threads);
        free(workers);
        return -1;
    }

    struct timespec wall_start, wall_end;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    for (int index = 0; index < concurrency; index++) {
        workers[index] = (concurrent_worker_args_t){
            .file_buf = file_buf,
            .file_len = file_len,
            .width = width,
            .iterations = iterations_per_thread,
            .latencies = latencies + ((size_t)index * (size_t)iterations_per_thread),
            .failed = 0,
        };
        if (pthread_create(&threads[index], NULL, run_concurrent_worker, &workers[index]) != 0) {
            clock_gettime(CLOCK_MONOTONIC, &wall_end);
            *wall_total_ns = timespec_diff_ns(&wall_start, &wall_end);
            for (int joined = 0; joined < index; joined++) {
                pthread_join(threads[joined], NULL);
            }
            free(threads);
            free(workers);
            return -1;
        }
    }

    int failed = 0;
    for (int index = 0; index < concurrency; index++) {
        pthread_join(threads[index], NULL);
        if (workers[index].failed) {
            failed = 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    *wall_total_ns = timespec_diff_ns(&wall_start, &wall_end);

    free(threads);
    free(workers);
    return failed ? -1 : 0;
}

static int cmp_i64(const void *lhs, const void *rhs) {
    const long long left = *(const long long *)lhs;
    const long long right = *(const long long *)rhs;
    return (left > right) - (left < right);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    if (VIPS_INIT(argv[0])) {
        fprintf(stderr, "Failed to init libvips\n");
        return 1;
    }

    const char *input_path = argv[1];
    const char *scenario = argv[2];
    int iterations = DEFAULT_ITERATIONS;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[i + 1]);
            if (iterations <= 0 || iterations > MAX_ITERATIONS) {
                iterations = DEFAULT_ITERATIONS;
            }
            break;
        }
    }

    FILE *f = fopen(input_path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open: %s\n", input_path);
        return 1;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 1;
    }
    long file_len = ftell(f);
    if (file_len <= 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return 1;
    }
    void *file_buf = malloc((size_t)file_len);
    if (!file_buf || fread(file_buf, 1, (size_t)file_len, f) != (size_t)file_len) {
        fprintf(stderr, "Failed to read file into buffer\n");
        fclose(f);
        free(file_buf);
        return 1;
    }
    fclose(f);

    int width = 400;
    int quality = 85;
    int concurrency = 0;
    int iterations_per_thread = 0;
    const char *format = "webp";

    int arg_start = 3;
    int arg_end = argc;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--iterations") == 0) {
            arg_end = i;
            break;
        }
    }

    if (strcmp(scenario, "thumbnail-bytes") == 0) {
        if (arg_start < arg_end) width = atoi(argv[arg_start]);
        if (width <= 0) width = 400;
    } else if (strcmp(scenario, "pipeline-bytes") == 0) {
        if (arg_start < arg_end) width = atoi(argv[arg_start]);
        if (arg_start + 1 < arg_end) quality = atoi(argv[arg_start + 1]);
        if (width <= 0) width = 800;
        if (quality <= 0 || quality > 100) quality = 85;
    } else if (strcmp(scenario, "concurrent") == 0) {
        if (arg_start < arg_end) width = atoi(argv[arg_start]);
        if (arg_start + 1 < arg_end) concurrency = atoi(argv[arg_start + 1]);
        if (arg_start + 2 < arg_end) iterations_per_thread = atoi(argv[arg_start + 2]);
        if (width <= 0) width = 400;
        if (concurrency <= 0) concurrency = 4;
        if (iterations_per_thread <= 0 || iterations_per_thread > MAX_ITERATIONS) {
            iterations_per_thread = 5;
        }
    } else if (strcmp(scenario, "large-upload") == 0) {
        if (arg_start < arg_end) width = atoi(argv[arg_start]);
        if (width <= 0) width = 400;
    } else if (strcmp(scenario, "workflow") == 0) {
        if (arg_start < arg_end) format = argv[arg_start];
        if (arg_start + 1 < arg_end) width = atoi(argv[arg_start + 1]);
        if (width <= 0) width = 400;
    } else {
        fprintf(stderr, "Unknown scenario: %s\n", scenario);
        print_usage(argv[0]);
        free(file_buf);
        return 1;
    }

    for (int i = 0; i < 3; i++) {
        int ret = -1;
        if (strcmp(scenario, "thumbnail-bytes") == 0 || strcmp(scenario, "large-upload") == 0) {
            ret = run_thumbnail_bytes(file_buf, (size_t)file_len, width);
        } else if (strcmp(scenario, "pipeline-bytes") == 0) {
            ret = run_pipeline_bytes(file_buf, (size_t)file_len, width, quality);
        } else if (strcmp(scenario, "concurrent") == 0) {
            long long warmup_wall = 0;
            long long *warmup_latencies =
                calloc((size_t)concurrency * (size_t)iterations_per_thread, sizeof(long long));
            if (!warmup_latencies) {
                free(file_buf);
                return 1;
            }
            ret = run_concurrent_scenario(
                file_buf,
                (size_t)file_len,
                width,
                concurrency,
                iterations_per_thread,
                warmup_latencies,
                &warmup_wall
            );
            free(warmup_latencies);
        } else if (strcmp(scenario, "workflow") == 0) {
            ret = run_workflow_web(file_buf, (size_t)file_len, width, format);
        }

        if (ret != 0) {
            fprintf(stderr, "Warmup failed for scenario '%s'\n", scenario);
            free(file_buf);
            return 1;
        }
    }

    int sample_count = iterations;
    if (strcmp(scenario, "concurrent") == 0) {
        sample_count = concurrency * iterations_per_thread;
    }

    long long *wall_ns = calloc((size_t)sample_count, sizeof(long long));
    if (!wall_ns) {
        free(file_buf);
        return 1;
    }

    long long wall_total_ns = 0;
    long peak_rss_kb = 0;
    long rss_before = current_resident_kb();

    if (strcmp(scenario, "concurrent") == 0) {
        if (run_concurrent_scenario(
                file_buf,
                (size_t)file_len,
                width,
                concurrency,
                iterations_per_thread,
                wall_ns,
                &wall_total_ns
            ) != 0) {
            fprintf(stderr, "Concurrent scenario failed\n");
            free(wall_ns);
            free(file_buf);
            return 1;
        }
        peak_rss_kb = current_resident_kb() - rss_before;
    } else {
        for (int i = 0; i < iterations; i++) {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);

            int ret = -1;
            if (strcmp(scenario, "thumbnail-bytes") == 0 || strcmp(scenario, "large-upload") == 0) {
                ret = run_thumbnail_bytes(file_buf, (size_t)file_len, width);
            } else if (strcmp(scenario, "pipeline-bytes") == 0) {
                ret = run_pipeline_bytes(file_buf, (size_t)file_len, width, quality);
            } else if (strcmp(scenario, "workflow") == 0) {
                ret = run_workflow_web(file_buf, (size_t)file_len, width, format);
            }

            clock_gettime(CLOCK_MONOTONIC, &t1);

            if (ret != 0) {
                fprintf(stderr, "Iteration %d failed\n", i);
                free(wall_ns);
                free(file_buf);
                return 1;
            }
            wall_ns[i] = timespec_diff_ns(&t0, &t1);
            wall_total_ns += wall_ns[i];
            long rss_now = current_resident_kb() - rss_before;
            if (rss_now > peak_rss_kb) {
                peak_rss_kb = rss_now;
            }
        }
    }

    qsort(wall_ns, (size_t)sample_count, sizeof(long long), cmp_i64);

    printf("{\"scenario\":\"%s\",\"iterations\":%d,\"wall_ns\":[", scenario,
           strcmp(scenario, "concurrent") == 0 ? sample_count : iterations);
    for (int i = 0; i < sample_count; i++) {
        if (i > 0) printf(",");
        printf("%lld", wall_ns[i]);
    }
    printf("],\"wall_total_ns\":%lld", wall_total_ns);
    printf(",\"peak_rss_kb\":%ld", peak_rss_kb < 0 ? 0 : peak_rss_kb);
    printf(",\"input_bytes\":%ld", file_len);
    printf(",\"width\":%d", width);
    if (strcmp(scenario, "concurrent") == 0) {
        printf(",\"concurrency\":%d", concurrency);
        printf(",\"iterations_per_thread\":%d", iterations_per_thread);
    }
    printf("}\n");

    free(wall_ns);
    free(file_buf);
    vips_shutdown();
    return 0;
}
