#include <stdio.h>
#include <stdbool.h>

#define NOB_STRIP_PREFIX
#include "nob.h"

#include "vulkan/vulkan.h"

#include "vulkan_globals.h"
#include "vulkan_createComputePipelines.h"


bool createComputePipeline_opts(CreateComputePipelineARGS args){
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {0};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.pNext = NULL;
    pipelineLayoutCreateInfo.flags = 0;
    pipelineLayoutCreateInfo.setLayoutCount = args.descriptorSetLayoutCount;
    pipelineLayoutCreateInfo.pSetLayouts = args.descriptorSetLayouts;

    if(args.pushConstantsSize > 0){
        VkPushConstantRange pushConstantRange = {0};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = args.pushConstantsSize;

        pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
        pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    }else{
        pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
        pipelineLayoutCreateInfo.pPushConstantRanges = NULL;
    }

    VkResult result = vkCreatePipelineLayout(device,&pipelineLayoutCreateInfo,NULL,args.pipelineLayoutOUT);
    if(result != VK_SUCCESS){
        printf("Couldn't create pipeline layout\n");
        return false;
    }

    
    VkComputePipelineCreateInfo computePipelineCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .stage = (VkPipelineShaderStageCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = args.computeShader,
            .pSpecializationInfo = NULL,
            .pName = "main",
        },
        .layout = *args.pipelineLayoutOUT,
    };

    result = vkCreateComputePipelines(device,NULL,1,&computePipelineCreateInfo,NULL,args.pipelineOUT);

    if(result != VK_SUCCESS){
        printf("ERROR: Couldn't create compute pipeline\n");
        return false;
    }

    return true;
}
