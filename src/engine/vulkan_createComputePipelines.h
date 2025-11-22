#ifndef TRIEX_VULKAN_CREATE_COMPUTE_PIPELINES
#define TRIEX_VULKAN_CREATE_COMPUTE_PIPELINES

#include <stdbool.h>
#include <vulkan/vulkan.h>

typedef struct {
    VkShaderModule computeShader;
    size_t pushConstantsSize;
    VkPipeline* pipelineOUT;
    VkPipelineLayout* pipelineLayoutOUT;
    size_t descriptorSetLayoutCount;
    VkDescriptorSetLayout* descriptorSetLayouts;
} CreateComputePipelineARGS;

bool createComputePipeline_opts(CreateComputePipelineARGS args);

#define createComputePipeline(computeShaderIn, pipelinePtrOut, pipelineLayoutPtrOut, ...) createComputePipeline_opts((CreateComputePipelineARGS){\
    .computeShader = computeShaderIn,\
    .pipelineOUT = pipelinePtrOut,\
    .pipelineLayoutOUT = pipelineLayoutPtrOut,\
    __VA_ARGS__})

#endif