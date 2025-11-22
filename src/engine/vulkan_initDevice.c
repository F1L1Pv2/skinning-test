#include <stdio.h>
#include <stdbool.h>

#define NOB_STRIP_PREFIX
#include "nob.h"

#include "vulkan/vulkan.h"

#include "vulkan_globals.h"
#include "vulkan_initDevice.h"
#include "vulkan_internal.h"

VkDevice device;
VkPhysicalDevice physicalDevice;
VkPhysicalDeviceLimits physicalDeviceLimits;
VkQueue computeQueue = NULL;
VkQueue graphicsQueue = NULL;
VkQueue presentQueue = NULL;
VkPhysicalDeviceMemoryProperties physicalMemoryProperties;
MultipleVkQueueFamilyProperties multipleQueueFamilyProperties;

typedef struct{
    VkPhysicalDevice* items;
    uint32_t count;
    size_t capacity;
} VkPhysicalDevices;

typedef struct{
    VkDeviceQueueCreateInfo* items;
    size_t count;
    size_t capacity;
} VkDeviceQueueCreateInfos;

typedef struct{
    float* items;
    size_t count;
    size_t capacity;
}   VkQueuePriorities;

typedef struct{
    VkQueuePriorities* items;
    size_t count;
    size_t capacity;
}   MultipleVkQueuePriorities;

bool initDevice(){

    VkPhysicalDevices physicalDevices = {0};

    vkEnumeratePhysicalDevices(instance, &physicalDevices.count, NULL);
    if (physicalDevices.count == 0) {
        printf("ERROR: No compatible GPU devices\n");
        return false;
    }

    da_resize(&physicalDevices, physicalDevices.count);
    vkEnumeratePhysicalDevices(instance, &physicalDevices.count, physicalDevices.items);

    physicalDevice = physicalDevices.items[0];

    if (physicalDevices.count > 1) {
        for (size_t i = 0; i < physicalDevices.count; i++) {
            VkPhysicalDeviceProperties properties;
            vkGetPhysicalDeviceProperties(physicalDevices.items[i], &properties);

            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physicalDevice = physicalDevices.items[i];
                break;
            }
        }
    }

    VkPhysicalDeviceProperties physicalDeviceProperties = {0};
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
    physicalDeviceLimits = physicalDeviceProperties.limits;

    printf("INFO: Chosen %s physical device\n", physicalDeviceProperties.deviceName);

    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalMemoryProperties);

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &multipleQueueFamilyProperties.count, NULL);
    da_resize(&multipleQueueFamilyProperties, multipleQueueFamilyProperties.count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &multipleQueueFamilyProperties.count, multipleQueueFamilyProperties.items);

    VkDeviceQueueCreateInfos queueCreateInfos = {0};

    MultipleVkQueuePriorities multipleVkQueuePriorities = {0};

    int computeQueueFamilyIndex = -1;
    int graphicsQueueFamilyIndex = -1;
    int presentQueueFamilyIndex = -1;

    for (size_t i = 0; i < multipleQueueFamilyProperties.count; i++) {
        if (graphicsQueueFamilyIndex != -1 && presentQueueFamilyIndex != -1 && computeQueueFamilyIndex != -1) break;

        if ((multipleQueueFamilyProperties.items[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && computeQueueFamilyIndex == -1) {
            computeQueueFamilyIndex = (int)i;
        }

        if ((multipleQueueFamilyProperties.items[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && graphicsQueueFamilyIndex == -1) {
            graphicsQueueFamilyIndex = (int)i;
        }

        if (presentQueueFamilyIndex == -1 && surface) {
            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, (uint32_t)i, surface, &presentSupport);
            if (presentSupport) presentQueueFamilyIndex = (int)i;
        }
    }

    if (graphicsQueueFamilyIndex == -1) {
        printf("ERROR: Couldn't find graphics queue\n");
        return false;
    }

    #define ADD_QUEUE_FAMILY(familyIndexVar, queueCountExpr, createInfoVar)                 \
        do {                                                                                \
            VkQueuePriorities tmpPriorities = {0};                                          \
            for (size_t q = 0; q < (queueCountExpr); q++) {                                 \
                da_append(&tmpPriorities, 1.0f);                                            \
            }                                                                               \
            da_append(&multipleVkQueuePriorities, tmpPriorities);                          \
                                                                                            \
            VkDeviceQueueCreateInfo createInfoVar = {0};                                    \
            createInfoVar.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;               \
            createInfoVar.pNext = NULL;                                                      \
            createInfoVar.flags = 0;                                                         \
            createInfoVar.queueFamilyIndex = (uint32_t)(familyIndexVar);                    \
            createInfoVar.queueCount = (uint32_t)(queueCountExpr);                          \
            createInfoVar.pQueuePriorities = multipleVkQueuePriorities.items[multipleVkQueuePriorities.count - 1].items; \
            da_append(&queueCreateInfos, createInfoVar);                                    \
        } while (0)

    // Add graphics queue family (always)
    ADD_QUEUE_FAMILY(graphicsQueueFamilyIndex, multipleQueueFamilyProperties.items[graphicsQueueFamilyIndex].queueCount, graphicsQueueCreateInfo);

    // Add present queue family if different than graphics
    if (presentQueueFamilyIndex != -1 && presentQueueFamilyIndex != graphicsQueueFamilyIndex) {
        ADD_QUEUE_FAMILY(presentQueueFamilyIndex, multipleQueueFamilyProperties.items[presentQueueFamilyIndex].queueCount, presentQueueCreateInfo);
    }

    // Add compute queue family if different from both graphics and present
    if (computeQueueFamilyIndex != -1 &&
        computeQueueFamilyIndex != graphicsQueueFamilyIndex &&
        computeQueueFamilyIndex != presentQueueFamilyIndex) {

        ADD_QUEUE_FAMILY(computeQueueFamilyIndex, multipleQueueFamilyProperties.items[computeQueueFamilyIndex].queueCount, computeQueueCreateInfo);
    }

    #undef ADD_QUEUE_FAMILY

    if (queueCreateInfos.count == 0) {
        printf("ERROR: Your physical device doesn't have any of the needed queues\n");
        // free any allocated priorities arrays before returning
        for (size_t i = 0; i < multipleVkQueuePriorities.count; i++) {
            da_free((multipleVkQueuePriorities.items[i]));
        }
        da_free(multipleVkQueuePriorities);
        return false;
    }

    VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeature = {0};
    dynamicRenderingFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    dynamicRenderingFeature.dynamicRendering = VK_TRUE;

    VkPhysicalDeviceDescriptorIndexingFeatures indexingFeatures = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
        .pNext = NULL,
        .runtimeDescriptorArray = VK_TRUE,
        .descriptorBindingPartiallyBound = VK_TRUE,
        .descriptorBindingVariableDescriptorCount = VK_TRUE,
    };

    dynamicRenderingFeature.pNext = &indexingFeatures;

    VkDeviceCreateInfo deviceInfo = {0};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext = &dynamicRenderingFeature;
    deviceInfo.flags = 0;
    deviceInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos.count;
    deviceInfo.pQueueCreateInfos = queueCreateInfos.items;
    const char* extensions[] = {
        "VK_KHR_swapchain",
        "VK_KHR_dynamic_rendering"
    };
    deviceInfo.enabledExtensionCount = ARRAY_LEN(extensions);
    deviceInfo.ppEnabledExtensionNames = extensions;
    deviceInfo.pEnabledFeatures = NULL;

    VkResult result = vkCreateDevice(physicalDevice, &deviceInfo, NULL, &device);
    if (result != VK_SUCCESS) {
        printf("ERROR: Couldn't create device (vkCreateDevice returned %d)\n", result);
        for (size_t i = 0; i < multipleVkQueuePriorities.count; i++) {
            da_free((multipleVkQueuePriorities.items[i]));
        }
        da_free(multipleVkQueuePriorities);
        da_free(queueCreateInfos);
        return false;
    }

    for (size_t i = 0; i < multipleVkQueuePriorities.count; i++) {
        da_free((multipleVkQueuePriorities.items[i]));
    }
    da_free(multipleVkQueuePriorities);
    da_free(queueCreateInfos);

    vkGetDeviceQueue(device, (uint32_t)graphicsQueueFamilyIndex, 0, &graphicsQueue);
    if (presentQueueFamilyIndex != -1) {
        vkGetDeviceQueue(device, (uint32_t)presentQueueFamilyIndex, 0, &presentQueue);
    } else {
        presentQueue = VK_NULL_HANDLE;
    }
    if (computeQueueFamilyIndex != -1) {
        vkGetDeviceQueue(device, (uint32_t)computeQueueFamilyIndex, 0, &computeQueue);
    } else {
        computeQueue = VK_NULL_HANDLE;
    }

    return true;
}