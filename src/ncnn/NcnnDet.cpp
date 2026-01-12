#define NCNNDET_EXPORTS // 添加这一行，将 API 切换为导出模式
#include "NcnnDet.h"

#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include "net.h"
#include "gpu.h" // 引入 GPU 支持

static int g_gpu_ref_count = 0;

class NcnnDet {
public:
    NcnnDet() = default;

    void Init(const std::string& param_path, float conf, float iou, bool use_gpu) {
        conf_threshold = conf;
        iou_threshold = iou;
        if (use_gpu) {
            int gpu_count = ncnn::get_gpu_count();
            if (gpu_count > 0) {
                if (g_gpu_ref_count == 0) ncnn::create_gpu_instance();
                g_gpu_ref_count++;
                use_gpu_initialized = true;
            } else {
                std::cout << "[Warn] No GPU found, fallback to CPU." << std::endl;
            }
        }

        net = std::make_unique<ncnn::Net>();
        net->opt.use_vulkan_compute = use_gpu_initialized;
        net->load_param(param_path.c_str());
        std::string bin_path = param_path;
        const std::string from_ext = ".param";
        const std::string to_ext = ".bin";
        size_t pos = bin_path.rfind(from_ext);
        if (pos != std::string::npos) {
            bin_path.replace(pos, from_ext.length(), to_ext);
        }
        net->load_model(bin_path.c_str());

        input_blob = "in0";
        output_blob = "out0";
    }

    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>>  operator()(cv::Mat& image) {
        return DetectObjects(image);
    }

    void setInputSize(int size) {
        input_size = size;
    }

    void setBlobNames(const std::string& input_name, const std::string& output_name) {
        input_blob = input_name;
        output_blob = output_name;
    }


private:
    std::unique_ptr<ncnn::Net> net;
    float conf_threshold = 0.3f, iou_threshold = 0.5f;
    int input_size = 640;
    bool use_gpu_initialized = false;
    std::string input_blob = "in0", output_blob = "out0";

    static ncnn::Mat Transpose2D(const ncnn::Mat& mat) {
        if (mat.dims != 2) return mat;
        ncnn::Mat transposed(mat.h, mat.w); // 注意顺序：w, h
        for (int i = 0; i < mat.h; ++i) {
            const float* src = mat.row(i);
            for (int j = 0; j < mat.w; ++j) {
                transposed.row(j)[i] = src[j];
            }
        }
        return transposed;
    }

    std::tuple<ncnn::Mat, float> PrepareInput(const cv::Mat& image) {
        int img_w = image.cols;
        int img_h = image.rows;

        float r = (std::min)((float)input_size / img_h, (float)input_size / img_w);
        int new_unpad_w = (int)(img_w * r);
        int new_unpad_h = (int)(img_h * r);

        cv::Mat resized;
        if (img_w != new_unpad_w || img_h != new_unpad_h) {
            cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h));
        } else {
            resized = image;
        }
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(resized.data, ncnn::Mat::PIXEL_BGR2RGB, new_unpad_w, new_unpad_h, input_size, input_size);
        cv::Mat input_img = cv::Mat(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114)); // 整张图先填114
        resized.copyTo(input_img(cv::Rect(0, 0, new_unpad_w, new_unpad_h)));   // 左上角贴原图

        ncnn::Mat in_net = ncnn::Mat::from_pixels(input_img.data, ncnn::Mat::PIXEL_BGR2RGB,input_size, input_size);
        const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
        in_net.substract_mean_normalize(nullptr, norm_vals);

        return {in_net, 1.0f / r};
    }

    static void NMS(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, const float iou_threshold, std::vector<int>& indices) {
        indices.clear();
        if (boxes.empty()) return;

        std::vector<std::pair<float, int>> score_index;
        for (int i = 0; i < boxes.size(); ++i) {
            score_index.emplace_back(scores[i], i);
        }

        std::sort(score_index.begin(), score_index.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

        std::vector<bool> suppressed(boxes.size(), false);
        for (size_t _i = 0; _i < score_index.size(); ++_i) {
            int i = score_index[_i].second;
            if (suppressed[i]) continue;
            indices.push_back(i);

            for (size_t _j = _i + 1; _j < score_index.size(); ++_j) {
                int j = score_index[_j].second;
                if (suppressed[j]) continue;

                float xx1 = (std::max)(boxes[i].x, boxes[j].x);
                float yy1 = (std::max)(boxes[i].y, boxes[j].y);
                float xx2 = (std::min)(boxes[i].x + boxes[i].width, boxes[j].x + boxes[j].width);
                float yy2 = (std::min)(boxes[i].y + boxes[i].height, boxes[j].y + boxes[j].height);

                float w = (std::max)(0.0f, xx2 - xx1);
                float h = (std::max)(0.0f, yy2 - yy1);
                float inter_area = w * h;

                if (inter_area == 0) continue;

                float area_i = boxes[i].width * boxes[i].height;
                float area_j = boxes[j].width * boxes[j].height;
                float union_area = area_i + area_j - inter_area;

                if (inter_area / union_area > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> DetectObjects(const cv::Mat& image) {
        ncnn::Mat in;
        float retio;
        std::tie(in, retio) = PrepareInput(image);
        ncnn::Extractor ex = net->create_extractor();
        ex.input(input_blob.c_str(), in);
        ncnn::Mat out;
        ex.extract(output_blob.c_str(), out);

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        boxes.reserve(boxes.size());
        scores.reserve(scores.size());
        class_ids.reserve(class_ids.size());

        ncnn::Mat T_out;
        if (out.dims == 2) {
            T_out = Transpose2D(out);
        } else if (out.dims == 3) {
            T_out = Transpose2D(out.channel(0));
        } else {
            throw std::runtime_error("Unsupported output dimensions");
        }

        for (int i = 0; i < T_out.h; ++i) {
            const float* current_anchor_data = T_out.row(i);

            float score_max = -1.f;
            int class_id = -1;
            for (int c = 4; c < T_out.w; ++c) {
                float s = current_anchor_data[c];
                if (s > score_max) {
                    score_max = s;
                    class_id = c - 4;
                }
            }

            if (score_max < conf_threshold) continue;

            float x, y, w, h;
            x = T_out.row(i)[0];
            y = T_out.row(i)[1];
            w = T_out.row(i)[2];
            h = T_out.row(i)[3];

            float x1 = (x - w * 0.5f) * retio;
            float y1 = (y - h * 0.5f) * retio;
            float w_scaled = w * retio;
            float h_scaled = h * retio;

            int left = (int)x1;
            int top = (int)y1;
            int width = (int)w_scaled;
            int height = (int)h_scaled;

            if (width <= 0 || height <= 0) continue;

            left = (std::max)(0, (std::min)(left, image.cols - 1));
            top = (std::max)(0, (std::min)(top, image.rows - 1));
            if (left + width > image.cols) width = image.cols - left;
            if (top + height > image.rows) height = image.rows - top;

            boxes.emplace_back(left, top, width, height);
            scores.push_back(score_max);
            class_ids.push_back(class_id);
        }

        std::vector<int> keep;
        NMS(boxes, scores, iou_threshold, keep);

        std::vector<cv::Rect> out_boxes;
        std::vector<float> out_scores;
        std::vector<int> out_class_ids;
        for (int idx : keep) {
            out_boxes.push_back(boxes[idx]);
            out_scores.push_back(scores[idx]);
            out_class_ids.push_back(class_ids[idx]);
        }

        return {out_boxes, out_scores, out_class_ids};
    }
};

extern "C" {
    NCNNDET_API void* CreateDetector() {
        try {
            return new NcnnDet();
        } catch (const std::exception& e) {
            std::cerr << "Error creating detector" <<e.what() << std::endl;
            return nullptr;
        }
    }

    NCNNDET_API void DestroyDetector(void* detector) {
        if (detector) {
            delete static_cast<NcnnDet*>(detector);
            ncnn::destroy_gpu_instance();
        }
    }

    NCNNDET_API void SetInputSize(void* detector, int input_size) {
        if (!detector) return;
        try {
            static_cast<NcnnDet*>(detector)->setInputSize(input_size);
        } catch (const std::exception& e) {
            std::cerr << "Error setting input size: " << e.what() << std::endl;
        }
    }

    NCNNDET_API void SetBlobNames(void* detector, const std::string& input_name, const std::string& output_name) {
        if (!detector) return;
        try {
            static_cast<NcnnDet*>(detector)->setBlobNames(input_name, output_name);
        } catch (const std::exception& e) {
            std::cerr << "Error setting blob names: " << e.what() << std::endl;
        }
    }

    NCNNDET_API bool InitDetector(void* detector, const char* param_path, float conf, float iou, bool use_gpu) {
        if (!detector || !param_path) {
            return false;
        }
        try {
            static_cast<NcnnDet*>(detector)->Init(std::string(param_path), conf, iou, use_gpu);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error initializing detector: " << e.what() << std::endl;
            return false;
        }
    }

    NCNNDET_API bool Detect(void* detector, unsigned char* image_data, int width, int height, int channels, float** out_boxes, float** out_scores, int** out_classes, int* out_count) {
        if (!detector || !image_data || !out_boxes || !out_scores || !out_classes || !out_count) return false;

        try {
            cv::Mat image(height, width, CV_8UC3, image_data);
            auto results = (*static_cast<NcnnDet*>(detector))(image);

            const auto& boxes = std::get<0>(results);
            const auto& scores = std::get<1>(results);
            const auto& classes = std::get<2>(results);

            *out_count = static_cast<int>(boxes.size());
            if (*out_count == 0) {
                *out_boxes = nullptr;
                *out_scores = nullptr;
                *out_classes = nullptr;
                return true;
            }

            *out_boxes = (float*)malloc(boxes.size() * 4 * sizeof(float));
            *out_scores = (float*)malloc(scores.size() * sizeof(float));
            *out_classes = (int*)malloc(classes.size() * sizeof(int));

            for (size_t i = 0; i < boxes.size(); i++) {
                (*out_boxes)[i * 4] = static_cast<float>(boxes[i].x);
                (*out_boxes)[i * 4 + 1] = static_cast<float>(boxes[i].y);
                (*out_boxes)[i * 4 + 2] = static_cast<float>(boxes[i].x + boxes[i].width);
                (*out_boxes)[i * 4 + 3] = static_cast<float>(boxes[i].y + boxes[i].height);
                (*out_scores)[i] = scores[i];
                (*out_classes)[i] = classes[i];
            }

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Detection error: " << e.what() << std::endl;
            *out_count = 0;
            return false;
        }
    }

    NCNNDET_API void ReleaseResults(float* boxes, float* scores, int* classes) {
        if (boxes) free(boxes);
        if (scores) free(scores);
        if (classes) free(classes);
    }
}