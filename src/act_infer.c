#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>

#define IMAGE_COUNT  (1 * 1 * 3 * 224 * 224)
#define STATE_COUNT  2
#define LATENT_COUNT 32
#define ACTION_COUNT (1 * 8 * 3)

static const OrtApi *ort = NULL;

static void check_status(OrtStatus *status, const char *where) {
    if (status != NULL) {
        fprintf(stderr, "%s failed: %s\n", where, ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        exit(1);
    }
}

static int read_file_f32(const char *path, float *buf, size_t count) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("failed to open: %s\n", path);
        return -1;
    }

    size_t n = fread(buf, sizeof(float), count, fp);
    fclose(fp);

    if (n != count) {
        printf("read count mismatch: %s, got=%zu, expect=%zu\n", path, n, count);
        return -1;
    }

    return 0;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <model.onnx> <image.bin> <state.bin> <latent.bin>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    const char *state_path = argv[3];
    const char *latent_path = argv[4];

    float *image = (float *)malloc(sizeof(float) * IMAGE_COUNT);
    float *state = (float *)malloc(sizeof(float) * STATE_COUNT);
    float *latent = (float *)malloc(sizeof(float) * LATENT_COUNT);

    if (!image || !state || !latent) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    if (read_file_f32(image_path, image, IMAGE_COUNT) != 0) return 1;
    if (read_file_f32(state_path, state, STATE_COUNT) != 0) return 1;
    if (read_file_f32(latent_path, latent, LATENT_COUNT) != 0) return 1;

    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort) {
        fprintf(stderr, "failed to get ORT API\n");
        return 1;
    }

    OrtEnv *env = NULL;
    OrtSessionOptions *session_options = NULL;
    OrtSession *session = NULL;
    OrtMemoryInfo *memory_info = NULL;
    OrtValue *input_tensors[3] = {NULL, NULL, NULL};
    OrtValue *output_tensor = NULL;

    const char *input_names[] = {"image", "state", "latent"};
    const char *output_names[] = {"action"};

    int64_t image_shape[] = {1, 1, 3, 224, 224};
    int64_t state_shape[] = {1, 2};
    int64_t latent_shape[] = {1, 32};

    check_status(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "act_infer", &env), "CreateEnv");
    check_status(ort->CreateSessionOptions(&session_options), "CreateSessionOptions");
    check_status(ort->SetIntraOpNumThreads(session_options, 1), "SetIntraOpNumThreads");
    check_status(ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC), "SetSessionGraphOptimizationLevel");
    check_status(ort->CreateSession(env, model_path, session_options, &session), "CreateSession");
    check_status(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info), "CreateCpuMemoryInfo");

    check_status(
        ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            image,
            sizeof(float) * IMAGE_COUNT,
            image_shape,
            5,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensors[0]),
        "CreateTensor(image)");

    check_status(
        ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            state,
            sizeof(float) * STATE_COUNT,
            state_shape,
            2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensors[1]),
        "CreateTensor(state)");

    check_status(
        ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            latent,
            sizeof(float) * LATENT_COUNT,
            latent_shape,
            2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensors[2]),
        "CreateTensor(latent)");

    check_status(
        ort->Run(
            session,
            NULL,
            input_names,
            (const OrtValue *const *)input_tensors,
            3,
            output_names,
            1,
            &output_tensor),
        "Run");

    float *action = NULL;
    check_status(ort->GetTensorMutableData(output_tensor, (void **)&action), "GetTensorMutableData");

    const float action_q01[2] = {-0.1f, 0.0f};
const float action_q99[2] = {0.2f, 0.2f};

float left_vel =
    ((action[0] + 1.0f) * 0.5f) * (action_q99[0] - action_q01[0]) + action_q01[0];
float right_vel =
    ((action[1] + 1.0f) * 0.5f) * (action_q99[1] - action_q01[1]) + action_q01[1];

printf("left_vel=%f\n", left_vel);
printf("right_vel=%f\n", right_vel);
printf("gripper_raw=%f\n", action[2]);
printf("direction=%s\n", left_vel < right_vel ? "left" : "right");

    if (output_tensor) ort->ReleaseValue(output_tensor);
    if (input_tensors[0]) ort->ReleaseValue(input_tensors[0]);
    if (input_tensors[1]) ort->ReleaseValue(input_tensors[1]);
    if (input_tensors[2]) ort->ReleaseValue(input_tensors[2]);
    if (memory_info) ort->ReleaseMemoryInfo(memory_info);
    if (session) ort->ReleaseSession(session);
    if (session_options) ort->ReleaseSessionOptions(session_options);
    if (env) ort->ReleaseEnv(env);


    free(image);
    free(state);
    free(latent);
    return 0;
}
