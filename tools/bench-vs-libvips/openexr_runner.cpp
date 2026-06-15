#include <sys/resource.h>
#include <time.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include <vips/vips.h>

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfIO.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfThreading.h>

namespace {

constexpr int kMaxIterations = 10000;
constexpr int kDefaultIterations = 50;

struct InputBlob {
    void* raw_buf = nullptr;
    size_t raw_len = 0;
    int width = 0;
    int height = 0;
    int bands = 0;
    VipsBandFormat format = VIPS_FORMAT_NOTSET;
};

long long timespec_diff_ns(const timespec& start, const timespec& end) {
    return static_cast<long long>(end.tv_sec - start.tv_sec) * 1000000000LL +
           static_cast<long long>(end.tv_nsec - start.tv_nsec);
}

void print_usage(const char* prog) {
    std::fprintf(
        stderr,
        "Usage: %s <input> save-exr --iterations N --threads N [--e2e]\n",
        prog);
}

std::unique_ptr<InputBlob> load_input_blob(const char* input) {
    VipsImage* src_img = vips_image_new_from_file(input, nullptr);
    if (!src_img) {
        std::fprintf(stderr, "Failed to load input image: %s\n", input);
        return nullptr;
    }

    auto input_blob = std::make_unique<InputBlob>();
    input_blob->raw_buf = vips_image_write_to_memory(src_img, &input_blob->raw_len);
    input_blob->width = vips_image_get_width(src_img);
    input_blob->height = vips_image_get_height(src_img);
    input_blob->bands = vips_image_get_bands(src_img);
    input_blob->format = vips_image_get_format(src_img);
    g_object_unref(src_img);

    if (!input_blob->raw_buf || input_blob->raw_len == 0) {
        std::fprintf(stderr, "Failed to preload input image to memory: %s\n", input);
        if (input_blob->raw_buf) {
            g_free(input_blob->raw_buf);
        }
        return nullptr;
    }

    return input_blob;
}

class VectorOStream final : public Imf::OStream {
public:
    explicit VectorOStream(const char* file_name) : Imf::OStream(file_name) {}

    void write(const char c[], int n) override {
        const auto end = position_ + static_cast<uint64_t>(n);
        if (end > data_.size()) {
            data_.resize(static_cast<size_t>(end));
        }
        std::memcpy(data_.data() + position_, c, static_cast<size_t>(n));
        position_ = end;
    }

    uint64_t tellp() override { return position_; }

    void seekp(uint64_t pos) override {
        position_ = pos;
        if (position_ > data_.size()) {
            data_.resize(static_cast<size_t>(position_));
        }
    }

    [[nodiscard]] size_t size() const { return data_.size(); }

private:
    std::vector<char> data_;
    uint64_t position_ = 0;
};

int run_save_exr_image(const InputBlob& input_blob, int threads) {
    if (input_blob.format != VIPS_FORMAT_FLOAT) {
        std::fprintf(stderr, "save-exr baseline expects an F32 input image\n");
        return -1;
    }

    const char* channel_names[4][4] = {
        {"Y", nullptr, nullptr, nullptr},
        {"Y", "A", nullptr, nullptr},
        {"R", "G", "B", nullptr},
        {"R", "G", "B", "A"},
    };

    if (input_blob.bands < 1 || input_blob.bands > 4) {
        std::fprintf(stderr, "save-exr baseline supports 1-4 float bands, got %d\n", input_blob.bands);
        return -1;
    }

    try {
        VectorOStream output_stream("save-exr-benchmark");
        Imf::Header header(input_blob.width, input_blob.height);
        auto& channels = header.channels();
        for (int band = 0; band < input_blob.bands; band++) {
            channels.insert(channel_names[input_blob.bands - 1][band], Imf::Channel(Imf::FLOAT));
        }

        Imf::FrameBuffer frame_buffer;
        auto* pixels = static_cast<float*>(input_blob.raw_buf);
        const size_t pixel_stride = sizeof(float) * static_cast<size_t>(input_blob.bands);
        const size_t row_stride = pixel_stride * static_cast<size_t>(input_blob.width);

        for (int band = 0; band < input_blob.bands; band++) {
            frame_buffer.insert(
                channel_names[input_blob.bands - 1][band],
                Imf::Slice(
                    Imf::FLOAT,
                    reinterpret_cast<char*>(pixels + band),
                    pixel_stride,
                    row_stride));
        }

        Imf::OutputFile output_file(output_stream, header, threads);
        output_file.setFrameBuffer(frame_buffer);
        output_file.writePixels(input_blob.height);
        return output_stream.size() > 0 ? 0 : -1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "OpenEXR encode failed: %s\n", error.what());
        return -1;
    }
}

int run_save_exr(const char* input, int threads, bool e2e, const InputBlob* preloaded_input) {
    if (e2e) {
        auto loaded = load_input_blob(input);
        if (!loaded) {
            return -1;
        }
        const int ret = run_save_exr_image(*loaded, threads);
        if (loaded->raw_buf) {
            g_free(loaded->raw_buf);
        }
        return ret;
    }

    if (!preloaded_input) {
        std::fprintf(stderr, "Missing preloaded EXR input\n");
        return -1;
    }
    return run_save_exr_image(*preloaded_input, threads);
}

}  // namespace

int main(int argc, char** argv) {
    if (VIPS_INIT(argv[0])) {
        vips_error_exit(nullptr);
    }

    if (argc < 3) {
        print_usage(argv[0]);
        vips_shutdown();
        return 1;
    }

    const char* input = argv[1];
    const char* op_name = argv[2];
    if (std::strcmp(op_name, "save-exr") != 0) {
        std::fprintf(stderr, "Unsupported operation for openexr-runner: %s\n", op_name);
        vips_shutdown();
        return 1;
    }

    int iterations = kDefaultIterations;
    int threads = 0;
    bool e2e = false;

    for (int i = 3; i < argc; i++) {
        if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[i + 1]);
            if (iterations <= 0 || iterations > kMaxIterations) {
                iterations = kDefaultIterations;
            }
            i++;
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[i + 1]);
            if (threads <= 0) {
                std::fprintf(stderr, "--threads requires a non-zero integer worker count\n");
                vips_shutdown();
                return 1;
            }
            i++;
        } else if (std::strcmp(argv[i], "--e2e") == 0) {
            e2e = true;
        }
    }

    if (threads == 0) {
        std::fprintf(stderr, "--threads is required\n");
        vips_shutdown();
        return 1;
    }

    Imf::setGlobalThreadCount(threads);

    std::unique_ptr<InputBlob> preloaded_input;
    if (!e2e) {
        preloaded_input = load_input_blob(input);
        if (!preloaded_input) {
            vips_shutdown();
            return 1;
        }
    }

    for (int i = 0; i < 3; i++) {
        if (run_save_exr(input, threads, e2e, preloaded_input.get()) != 0) {
            if (preloaded_input && preloaded_input->raw_buf) {
                g_free(preloaded_input->raw_buf);
            }
            vips_shutdown();
            return 1;
        }
    }

    rusage ru_before {};
    rusage ru_after {};
    getrusage(RUSAGE_SELF, &ru_before);

    auto* wall_ns = static_cast<long long*>(std::malloc(sizeof(long long) * static_cast<size_t>(iterations)));
    if (!wall_ns) {
        if (preloaded_input && preloaded_input->raw_buf) {
            g_free(preloaded_input->raw_buf);
        }
        vips_shutdown();
        return 1;
    }

    timespec t0 {};
    timespec t1 {};
    for (int i = 0; i < iterations; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        const int ret = run_save_exr(input, threads, e2e, preloaded_input.get());
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (ret != 0) {
            std::fprintf(stderr, "Operation failed at iteration %d\n", i);
            std::free(wall_ns);
            if (preloaded_input && preloaded_input->raw_buf) {
                g_free(preloaded_input->raw_buf);
            }
            vips_shutdown();
            return 1;
        }
        wall_ns[i] = timespec_diff_ns(t0, t1);
    }

    getrusage(RUSAGE_SELF, &ru_after);

    std::printf("{\n");
    std::printf("  \"backend\": \"openexr\",\n");
    std::printf("  \"version\": \"%d.%d.%d\",\n", OPENEXR_VERSION_MAJOR, OPENEXR_VERSION_MINOR, OPENEXR_VERSION_PATCH);
    std::printf("  \"input\": \"%s\",\n", input);
    std::printf("  \"operation\": \"%s\",\n", op_name);
    std::printf("  \"iterations\": %d,\n", iterations);
    std::printf("  \"wall_ns\": [");
    for (int i = 0; i < iterations; i++) {
        std::printf("%lld%s", wall_ns[i], i < iterations - 1 ? "," : "");
    }
    std::printf("],\n");
    std::printf("  \"peak_rss_kb\": %ld,\n", ru_after.ru_maxrss / 1024);
    std::printf("  \"minor_faults\": %ld,\n", ru_after.ru_minflt - ru_before.ru_minflt);
    std::printf("  \"major_faults\": %ld,\n", ru_after.ru_majflt - ru_before.ru_majflt);
    std::printf("  \"vol_ctx_switches\": %ld,\n", ru_after.ru_nvcsw - ru_before.ru_nvcsw);
    std::printf("  \"invol_ctx_switches\": %ld\n", ru_after.ru_nivcsw - ru_before.ru_nivcsw);
    std::printf("}\n");

    std::free(wall_ns);
    if (preloaded_input && preloaded_input->raw_buf) {
        g_free(preloaded_input->raw_buf);
    }
    vips_shutdown();
    return 0;
}
