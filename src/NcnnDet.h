#include <stdbool.h>
#include <stdlib.h>
#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #ifdef NCNNDET_EXPORTS
        #define NCNNDET_API __declspec(dllexport)
    #else
        #define NCNNDET_API __declspec(dllimport)
    #endif
#else
    #if defined(__GNUC__)
        #define NCNNDET_API __attribute__((visibility("default")))
    #else
        #define NCNNDET_API
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

    NCNNDET_API void* CreateDetector();
    NCNNDET_API void DestroyDetector(void* detector);
    NCNNDET_API void SetInputSize(void* detector, int input_size);
    NCNNDET_API bool InitDetector(void* detector, const char* param_path, float conf_threshold, float iou_threshold, bool use_gpu);
    NCNNDET_API bool Detect(void* detector, unsigned char* image_data, int width, int height, int channels,
        float** out_boxes, float** out_scores, int** out_classes, int* out_count);
    NCNNDET_API void ReleaseResults(float* boxes, float* scores, int* classes);
    NCNNDET_API void SetBlobNames(void* detector, const char* input_name, const char* output_name);

#ifdef __cplusplus
}
#endif
