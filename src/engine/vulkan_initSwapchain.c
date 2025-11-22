#include <stdio.h>
#include <stdbool.h>

#define NOB_STRIP_PREFIX
#include "nob.h"

#include "vulkan/vulkan.h"

#include "vulkan_globals.h"
#include "vulkan_initSwapchain.h"
#include "vulkan_images.h"
#include "vulkan_internal.h"
#include "vulkan_helpers.h"

VkSwapchainKHR swapchain;
VkFormat swapchainImageFormat;
VkExtent2D swapchainExtent;
VkImages swapchainImages;
VkImageViews swapchainImageViews;
VkFormat swapchainDepthFormat = VK_FORMAT_UNDEFINED;
VkImages swapchainDepthImages;
VkImageViews swapchainDepthImageViews;
VkDeviceMemories swapchainDepthImageMemories;
bool swapchainHasDepth = false;
bool swapchainHasStencil = false;

typedef struct {
    VkSurfaceFormatKHR* items;
    uint32_t count;
    size_t capacity;
} VkSurfaceFormats;

uint32_t clamp_uint32(uint32_t a, uint32_t min, uint32_t max){
    if(a < min) return min;
    if(a > max) return max;
    return a;
}

static bool findSupportedFormat(VkFormat* candidates_ptr, size_t candidates_count, VkImageTiling tiling, VkFormatFeatureFlags features, VkFormat* out) {
    if(out == NULL) return false;
    for(size_t i = 0; i < candidates_count; i++){
        VkFormat format = candidates_ptr[i];
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            *out = format;
            return true;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            *out = format;
            return true;
        }
    }

    return false;
}

static bool findDepthFormat(VkFormat* out) {
    return findSupportedFormat(
        (VkFormat*)((VkFormat[]){VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT}),
        3,
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
        out
    );
}

static bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

bool initSwapchain(){
    VkSwapchainCreateInfoKHR swapchainCreateInfoKHR = {0};
    swapchainCreateInfoKHR.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainCreateInfoKHR.pNext = NULL;
    swapchainCreateInfoKHR.flags = 0;
    swapchainCreateInfoKHR.surface = surface;

    VkSurfaceCapabilitiesKHR surfaceCapabilitiesKHR = {};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCapabilitiesKHR);

    VkSurfaceFormats surfaceFormats = {0};
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &surfaceFormats.count, NULL);
    da_resize(&surfaceFormats, surfaceFormats.count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &surfaceFormats.count, surfaceFormats.items);

    VkSurfaceFormatKHR surfaceFormat = surfaceFormats.items[0];
    bool foundFormat = false;
    for(size_t i = 0; i < surfaceFormats.count; i++){
        VkSurfaceFormatKHR availableFormat = surfaceFormats.items[i];
        if(availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR){
            surfaceFormat = availableFormat;
            foundFormat = true;
            break;
        }
    }

    if(foundFormat == false){
        printf("ERROR: Couldn't find needed image format\n");
        da_free(surfaceFormats);
        return false;
    }

    if(surfaceCapabilitiesKHR.currentExtent.width == UINT32_MAX){
        VkExtent2D extent = {
            clamp_uint32(640,surfaceCapabilitiesKHR.minImageExtent.width,surfaceCapabilitiesKHR.maxImageExtent.width),
            clamp_uint32(480,surfaceCapabilitiesKHR.minImageExtent.height,surfaceCapabilitiesKHR.maxImageExtent.height),
        };
        swapchainCreateInfoKHR.imageExtent = extent;
    }else{
        swapchainCreateInfoKHR.imageExtent = surfaceCapabilitiesKHR.currentExtent;
    }
    swapchainExtent = swapchainCreateInfoKHR.imageExtent;

    swapchainCreateInfoKHR.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t imageCount = surfaceCapabilitiesKHR.minImageCount + 1;
    if(surfaceCapabilitiesKHR.maxImageCount > 0 && imageCount > surfaceCapabilitiesKHR.maxImageCount){
        imageCount = surfaceCapabilitiesKHR.maxImageCount;
    }

    swapchainCreateInfoKHR.minImageCount = imageCount;
    swapchainCreateInfoKHR.imageFormat = surfaceFormat.format;
    swapchainImageFormat = surfaceFormat.format;
    swapchainCreateInfoKHR.imageColorSpace = surfaceFormat.colorSpace;
    swapchainCreateInfoKHR.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainCreateInfoKHR.preTransform = surfaceCapabilitiesKHR.currentTransform;
    swapchainCreateInfoKHR.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainCreateInfoKHR.imageArrayLayers = 1;

    VkResult result = vkCreateSwapchainKHR(device, &swapchainCreateInfoKHR, NULL, &swapchain);

    if(result != VK_SUCCESS){
        printf("ERROR: Couldn't create swapchain\n");
        da_free(surfaceFormats);
        return false;
    }

    da_free(surfaceFormats);


    vkGetSwapchainImagesKHR(device, swapchain, &swapchainImages.count,NULL);
    if(swapchainImages.count == 0) {
        printf("ERROR: Couldn't get swapchain images\n");
        return false;
    }

    da_resize(&swapchainImages,swapchainImages.count);
    vkGetSwapchainImagesKHR(device, swapchain, &swapchainImages.count,swapchainImages.items);

    da_resize(&swapchainImageViews,swapchainImages.count);

    for(int i = 0; i < swapchainImageViews.count; i++){
        if(!vkCreateImageViewEX(device, swapchainImages.items[i],swapchainImageFormat,VK_IMAGE_ASPECT_COLOR_BIT,&swapchainImageViews.items[i])) return false;
    }

    if(swapchainHasDepth){
        if(swapchainDepthFormat == VK_FORMAT_UNDEFINED){
            if(!findDepthFormat(&swapchainDepthFormat)) return false;
        }

        da_resize(&swapchainDepthImages,swapchainImages.count);
        da_resize(&swapchainDepthImageMemories, swapchainDepthImages.count);
        for(int i = 0; i < swapchainDepthImages.count;i++){
            if(!vkCreateImageEX(device, swapchainExtent.width,swapchainExtent.height, swapchainDepthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &swapchainDepthImages.items[i], &swapchainDepthImageMemories.items[i])) return false;
        }

        VkCommandBuffer tempCmd = vkCmdBeginSingleTime();

        swapchainHasStencil = hasStencilComponent(swapchainDepthFormat);

        for(size_t i = 0; i < swapchainDepthImages.count; i++){
            VkImageMemoryBarrier barrier = {0};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.pNext = NULL;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = swapchainHasStencil ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = swapchainDepthImages.items[i];
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(
                tempCmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                0,
                0, NULL,
                0, NULL,
                1, &barrier
            );
        }

        vkCmdEndSingleTime(tempCmd);

        da_resize(&swapchainDepthImageViews,swapchainDepthImages.count);
        for(int i = 0; i < swapchainDepthImageViews.count; i++){
            if(!vkCreateImageViewEX(device, swapchainDepthImages.items[i],swapchainDepthFormat,VK_IMAGE_ASPECT_DEPTH_BIT,&swapchainDepthImageViews.items[i])) return false;
        }
    }

    return true;
}