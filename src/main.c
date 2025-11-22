#include <stdio.h>
#include <string.h>

#include "engine/vulkan_simple.h"
#include "cgltf.h"
#include <assert.h>
#include <math.h>

#define PI 3.14159265359

typedef struct
{
    float pos_x;
    float pos_y;
    float pos_z;
    float normal_x;
    float normal_y;
    float normal_z;
    uint32_t joint_1;
    uint32_t joint_2;
    uint32_t joint_3;
    uint32_t joint_4;
    float weight_1;
    float weight_2;
    float weight_3;
    float weight_4;
} Vertex;

typedef struct{
    Vertex* items;
    size_t count;
    size_t capacity;
} Vertices;

typedef struct {
    uint32_t* items;
    size_t count;
    size_t capacity;
} Indices;

typedef struct{
    float proj[16];
    float view[16];
    float model[16];
    float invProj[16];
} Pcs;

typedef struct{
    VkDescriptorSet* items;
    size_t count;
    size_t capacity;
} VkDescriptorSets;

#define REAL_MOD(a, m) ( ((a) % (m) + (m)) % (m) )

#define FA_REALLOC(optr, osize, new_size) realloc(optr, new_size)
#define fa_reserve(da, extra) \
   do {\
      if((da)->count + extra >= (da)->capacity) {\
          void* _da_old_ptr;\
          size_t _da_old_capacity = (da)->capacity;\
          (void)_da_old_capacity;\
          (void)_da_old_ptr;\
          (da)->capacity = (da)->capacity*2+extra;\
          _da_old_ptr = (da)->items;\
          (da)->items = FA_REALLOC(_da_old_ptr, _da_old_capacity*sizeof(*(da)->items), (da)->capacity*sizeof(*(da)->items));\
          assert((da)->items && "Ran out of memory");\
      }\
   } while(0)
#define fa_push(da, value) \
   do {\
        fa_reserve(da, 1);\
        (da)->items[(da)->count++]=value;\
   } while(0)

void mat4_mul(const float a[16], const float b[16], float out[16])
{
    // Load A rows
    const float a00 = a[ 0], a01 = a[ 1], a02 = a[ 2], a03 = a[ 3];
    const float a10 = a[ 4], a11 = a[ 5], a12 = a[ 6], a13 = a[ 7];
    const float a20 = a[ 8], a21 = a[ 9], a22 = a[10], a23 = a[11];
    const float a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

    // Load B columns (still row-major, but accessed column-wise)
    const float b00 = b[ 0], b01 = b[ 1], b02 = b[ 2], b03 = b[ 3];
    const float b10 = b[ 4], b11 = b[ 5], b12 = b[ 6], b13 = b[ 7];
    const float b20 = b[ 8], b21 = b[ 9], b22 = b[10], b23 = b[11];
    const float b30 = b[12], b31 = b[13], b32 = b[14], b33 = b[15];

    // Row 0
    out[ 0] = a00*b00 + a01*b10 + a02*b20 + a03*b30;
    out[ 1] = a00*b01 + a01*b11 + a02*b21 + a03*b31;
    out[ 2] = a00*b02 + a01*b12 + a02*b22 + a03*b32;
    out[ 3] = a00*b03 + a01*b13 + a02*b23 + a03*b33;

    // Row 1
    out[ 4] = a10*b00 + a11*b10 + a12*b20 + a13*b30;
    out[ 5] = a10*b01 + a11*b11 + a12*b21 + a13*b31;
    out[ 6] = a10*b02 + a11*b12 + a12*b22 + a13*b32;
    out[ 7] = a10*b03 + a11*b13 + a12*b23 + a13*b33;

    // Row 2
    out[ 8] = a20*b00 + a21*b10 + a22*b20 + a23*b30;
    out[ 9] = a20*b01 + a21*b11 + a22*b21 + a23*b31;
    out[10] = a20*b02 + a21*b12 + a22*b22 + a23*b32;
    out[11] = a20*b03 + a21*b13 + a22*b23 + a23*b33;

    // Row 3
    out[12] = a30*b00 + a31*b10 + a32*b20 + a33*b30;
    out[13] = a30*b01 + a31*b11 + a32*b21 + a33*b31;
    out[14] = a30*b02 + a31*b12 + a32*b22 + a33*b32;
    out[15] = a30*b03 + a31*b13 + a32*b23 + a33*b33;
}

typedef struct{
    float data[16];
} Mat4;

typedef struct{
    Mat4* items;
    size_t count;
    size_t capacity;
} Mat4s;

static void quat_normalize(float q[4]) {
    float len = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (len > 0.0f) {
        float inv = 1.0f / len;
        q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
    }
}

static void quat_slerp(float out[4], const float a[4], const float b[4], float t)
{
    float cos_ = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];

    float bb[4] = {b[0], b[1], b[2], b[3]};

    if (cos_ < 0.0f) {
        cos_ = -cos_;
        bb[0] = -bb[0];
        bb[1] = -bb[1];
        bb[2] = -bb[2];
        bb[3] = -bb[3];
    }

    float k0, k1;

    if (cos_ > 0.9995f) {
        // Very close: use lerp
        k0 = 1.0f - t;
        k1 = t;
    } else {
        float sin_ = sqrtf(1.0f - cos_ * cos_);
        float angle = atan2f(sin_, cos_);
        k0 = sinf((1.0f - t) * angle) / sin_;
        k1 = sinf(t * angle) / sin_;
    }

    out[0] = a[0] * k0 + bb[0] * k1;
    out[1] = a[1] * k0 + bb[1] * k1;
    out[2] = a[2] * k0 + bb[2] * k1;
    out[3] = a[3] * k0 + bb[3] * k1;

    quat_normalize(out);
}

// ----------------- Keyframe search -----------------

static void find_keyframes(
    const cgltf_accessor* input,
    float time,
    size_t* k0,
    size_t* k1,
    float* t
) {
    size_t count = input->count;

    float t0 = 0.0f;
    float t1 = 0.0f;

    // Read first timestamp
    cgltf_accessor_read_float(input, 0, &t0, 1);

    // Before first frame
    if (time <= t0) {
        *k0 = *k1 = 0;
        *t = 0.0f;
        return;
    }

    // Search for interval
    for (size_t i = 0; i < count - 1; i++) {
        cgltf_accessor_read_float(input, i,     &t0, 1);
        cgltf_accessor_read_float(input, i + 1, &t1, 1);

        if (time >= t0 && time <= t1) {
            *k0 = i;
            *k1 = i + 1;
            *t = (time - t0) / (t1 - t0);
            return;
        }
    }

    // After last frame
    *k0 = *k1 = count - 1;
    *t = 0.0f;
}

// ----------------- Main animation sampler -----------------

void sample_animation(cgltf_animation* anim, float time, int loop)
{
    // Compute duration for looping
    float duration = 0.0f;
    for (size_t i = 0; i < anim->channels_count; i++) {
        float max_time = anim->channels[i].sampler->input->max[0];
        if (max_time > duration) duration = max_time;
    }

    if (loop && duration > 0.0f)
        time = fmodf(time, duration);

    for (size_t c = 0; c < anim->channels_count; c++) {

        cgltf_animation_channel* ch = &anim->channels[c];
        cgltf_animation_sampler* samp = ch->sampler;
        cgltf_node* node = ch->target_node;

        if (!node) continue;

        size_t k0, k1;
        float lerp_t;
        find_keyframes(samp->input, time, &k0, &k1, &lerp_t);

        // Evaluate based on channel target
        switch (ch->target_path) {
        case cgltf_animation_path_type_translation: {
            float v0[3], v1[3];
            cgltf_accessor_read_float(samp->output, k0, v0, 3);
            cgltf_accessor_read_float(samp->output, k1, v1, 3);
            for (int i = 0; i < 3; i++)
                node->translation[i] = v0[i] * (1 - lerp_t) + v1[i] * lerp_t;
        } break;

        case cgltf_animation_path_type_scale: {
            float v0[3], v1[3];
            cgltf_accessor_read_float(samp->output, k0, v0, 3);
            cgltf_accessor_read_float(samp->output, k1, v1, 3);
            for (int i = 0; i < 3; i++)
                node->scale[i] = v0[i] * (1 - lerp_t) + v1[i] * lerp_t;
        } break;

        case cgltf_animation_path_type_rotation: {
            float q0[4], q1[4], out[4];
            cgltf_accessor_read_float(samp->output, k0, q0, 4);
            cgltf_accessor_read_float(samp->output, k1, q1, 4);
            quat_slerp(out, q0, q1, lerp_t);
            for (int i = 0; i < 4; i++)
                node->rotation[i] = out[i];
        } break;

        default:
            break;
        }
    }
}

int main(){
    cgltf_options options = {0};
    cgltf_data* data = NULL;
    cgltf_result result = cgltf_parse_file(&options, "assets/mixamo-2.glb", &data);
    if (result != cgltf_result_success) return 1;
    result = cgltf_load_buffers(&options, data, "assets/mixamo-2.glb");
    if (result != cgltf_result_success) return 1;
    result = cgltf_validate(data);
    if (result != cgltf_result_success) return 1;

    assert(data->skins_count == 1);
    cgltf_skin* skin = &data->skins[0];
    cgltf_node* skinned_node = NULL;
    for (size_t i = 0; i < data->nodes_count; i++) {
        if (data->nodes[i].skin == skin) {
            skinned_node = &data->nodes[i];
            break;
        }
    }
    assert(skinned_node && "Skin has no node");

    assert(skinned_node->mesh && "Skin node has no mesh");
    cgltf_mesh* mesh = skinned_node->mesh;

    cgltf_primitive* prim = &mesh->primitives[0];
    Vertices vertices = {0};

    cgltf_accessor* pos_acc = NULL;
    cgltf_accessor* nor_acc = NULL;
    cgltf_accessor* joints_acc = NULL;
    cgltf_accessor* weights_acc = NULL;

    for (size_t i = 0; i < prim->attributes_count; i++) {
        cgltf_attribute* attr = &prim->attributes[i];

        if (attr->type == cgltf_attribute_type_position) pos_acc = attr->data;
        if (attr->type == cgltf_attribute_type_normal)   nor_acc = attr->data;
        if (attr->type == cgltf_attribute_type_joints)   joints_acc = attr->data;
        if (attr->type == cgltf_attribute_type_weights)   weights_acc = attr->data;
    }

    assert(pos_acc);
    assert(nor_acc);
    assert(joints_acc);
    assert(weights_acc);

    size_t vcount = pos_acc->count;
    assert(nor_acc->count == vcount);

    for (size_t i = 0; i < vcount; i++) {
        float p[3], n[3], w[4];
        uint32_t j[4];
        cgltf_accessor_read_float(pos_acc, i, p, 3);
        cgltf_accessor_read_float(nor_acc, i, n, 3);
        cgltf_accessor_read_uint(joints_acc, i, j, 4);
        cgltf_accessor_read_float(weights_acc, i, w, 4);

        Vertex v = {
            p[0], p[1], p[2],
            n[0], n[1], n[2],
            j[0], j[1], j[2], j[3],
            w[0], w[1], w[2], w[3],
        };
        fa_push(&vertices, v);
    }

    Indices indices = {0};
    cgltf_accessor* idx_acc = prim->indices;
    assert(idx_acc);

    size_t icount = idx_acc->count;

    for (size_t i = 0; i < icount; i++) {
        uint32_t idx = 0;
        cgltf_accessor_read_uint(idx_acc, i, &idx, 1);
        fa_push(&indices, idx);
    }

    printf("Loaded %zu vertices and %zu indices.\n",
       vertices.count, indices.count);

    Mat4s inverse_bind_matrices = {0};
    //cgltf_node_transform_local()
    for(size_t i = 0; i < skin->inverse_bind_matrices->count; i++){
        Mat4 mat4 = {0};
        cgltf_accessor_read_float(skin->inverse_bind_matrices, i, mat4.data, 16);
        fa_push(&inverse_bind_matrices, mat4);
    }

    assert(data->animations_count > 0);
    cgltf_animation* anim = &data->animations[0];
    sample_animation(anim, 0.0f, 1); // load first keyframe

    Mat4s combined_joint_matrices = {0};
    for(size_t i = 0; i < inverse_bind_matrices.count; i++){
        Mat4 joint_anim_mat4 = {0};
        cgltf_node_transform_world(skin->joints[i], joint_anim_mat4.data);
        Mat4 mat4 = {0};
        mat4_mul(inverse_bind_matrices.items[i].data, joint_anim_mat4.data, mat4.data);
        fa_push(&combined_joint_matrices, mat4);
    }

    // return 0;
    vulkan_init_with_window_and_depth_buffer("Skinning", 640, 480);

    VkCommandBuffer cmd;
    if(vkAllocateCommandBuffers(device,&(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    },&cmd) != VK_SUCCESS) return 1;

    VkShaderModule vertexShader;
    const char* vertexShaderSrc =
        "#version 450\n"
        "layout(std430,binding=0)readonly buffer J{mat4 m[];}j;"
        "layout(push_constant)uniform PC{mat4 p;mat4 v;mat4 M;mat4 iP;}pc;"
        "layout(location=0)in vec3 P;"
        "layout(location=1)in vec3 N;"
        "layout(location=2)in uvec4 JI;"
        "layout(location=3)in vec4 W;"
        "layout(location=0)out vec3 C;"
        "layout(location=1)out vec3 ON;"
        "layout(location=2)out vec3 OW;"
        "void main(){"
        "mat4 S=W.x*j.m[JI.x]+W.y*j.m[JI.y]+W.z*j.m[JI.z]+W.w*j.m[JI.w];"
        "vec4 WP=pc.M*S*vec4(P,1);"
        "OW=WP.xyz;"
        "ON=mat3(pc.M*S)*N;"
        "gl_Position=pc.p*pc.v*WP;"
        "C=vec3(1);"
        "}";

    if(!vkCompileShader(device, vertexShaderSrc, shaderc_vertex_shader,&vertexShader)) return 1;

    VkShaderModule fragmentShader;
    const char* fragmentShaderSrc =
        "#version 450\n"
        "layout(location=0)out vec4 OC;"
        "layout(location=0)in vec3 C;"
        "layout(location=1)in vec3 N;"
        "layout(location=2)in vec3 W;"
        "layout(push_constant)uniform PC{mat4 p;mat4 v;mat4 M;mat4 iP;}pc;"
        "void main(){"
        "vec3 n=normalize(N);"
        "vec3 CP=inverse(pc.v)[3].xyz;"
        "vec3 L=normalize(CP-W);"
        "float d=max(dot(n,L),0);"
        "OC=vec4(C*d,1);"
        "}";
    if(!vkCompileShader(device, fragmentShaderSrc, shaderc_fragment_shader,&fragmentShader)) return 1;

    VkDescriptorSetLayout jointMatricesDescriptorSetLayout = {0};
    vkCreateDescriptorSetLayout(device, &(VkDescriptorSetLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .bindingCount = 1,
        .pBindings = &(VkDescriptorSetLayoutBinding){
            .binding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT
        },
    }, NULL, &jointMatricesDescriptorSetLayout);

    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    if(!vkCreateGraphicPipeline(
        vertexShader, fragmentShader,
        &pipeline,
        &pipelineLayout,
        swapchainImageFormat,
        .pushConstantsSize = sizeof(Pcs),
        .vertexSize = sizeof(Vertex),
        .vertexInputAttributeDescriptionsCount = 4,
        .vertexInputAttributeDescriptions = (VkVertexInputAttributeDescription*)(VkVertexInputAttributeDescription[]){
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex,pos_x),
            },
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex,normal_x),
            },
            {
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32A32_UINT,
                .offset = offsetof(Vertex,joint_1),
            },
            {
                .location = 3,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = offsetof(Vertex,weight_1),
            },
        },
        .descriptorSetLayoutCount = 1,
        .descriptorSetLayouts = &jointMatricesDescriptorSetLayout,
        .depthTest = true,
        .culling = true,
        .outDepthFormat = swapchainDepthFormat,
    )) return 1;

    VkFence renderingFence;
    if(vkCreateFence(device,
        &(VkFenceCreateInfo){
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT 
        },
        NULL,
        &renderingFence
    ) != VK_SUCCESS) return 1;

    VkSemaphore swapchainHasImageSemaphore;
    if(vkCreateSemaphore(device, &(VkSemaphoreCreateInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}, NULL, &swapchainHasImageSemaphore) != VK_SUCCESS) return 1;
    VkSemaphore readyToSwapYourChainSemaphore;
    if(vkCreateSemaphore(device, &(VkSemaphoreCreateInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}, NULL, &readyToSwapYourChainSemaphore) != VK_SUCCESS) return 1;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    void* vertexBufferMapped;
    if(!vkCreateBufferEX(device, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, vertices.count*sizeof(*vertices.items), &vertexBuffer, &vertexBufferMemory)) return 1;
    if(vkMapMemory(device, vertexBufferMemory, 0, vertices.count*sizeof(*vertices.items), 0, &vertexBufferMapped) != VK_SUCCESS) return 1;
    memcpy(vertexBufferMapped, vertices.items, vertices.count*sizeof(*vertices.items));

    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    void* indexBufferMapped;
    if(!vkCreateBufferEX(device, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, indices.count*sizeof(*indices.items), &indexBuffer, &indexBufferMemory)) return 1;
    if(vkMapMemory(device, indexBufferMemory, 0, indices.count*sizeof(*indices.items), 0, &indexBufferMapped) != VK_SUCCESS) return 1;
    memcpy(indexBufferMapped, indices.items, indices.count*sizeof(*indices.items));

    VkBuffer jointsBuffer;
    VkDeviceMemory jointsBufferMemory;
    void* jointsBufferMapped;
    if(!vkCreateBufferEX(device, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, combined_joint_matrices.count*sizeof(*combined_joint_matrices.items), &jointsBuffer, &jointsBufferMemory)) return 1;
    if(vkMapMemory(device, jointsBufferMemory, 0, combined_joint_matrices.count*sizeof(*combined_joint_matrices.items), 0, &jointsBufferMapped) != VK_SUCCESS) return 1;
    memcpy(jointsBufferMapped, combined_joint_matrices.items, combined_joint_matrices.count*sizeof(*combined_joint_matrices.items));
    VkDescriptorSet jointsDescriptorSet;
    if(vkAllocateDescriptorSets(device, &(VkDescriptorSetAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = NULL,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &jointMatricesDescriptorSetLayout,
    }, &jointsDescriptorSet) != VK_SUCCESS) return 1;

    vkUpdateDescriptorSets(device, 1, &(VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = NULL,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .dstArrayElement = 0,
        .dstBinding = 0,
        .dstSet = jointsDescriptorSet,
        .pBufferInfo = &(VkDescriptorBufferInfo){
            .buffer = jointsBuffer,
            .offset = 0,
            .range = combined_joint_matrices.count*sizeof(*combined_joint_matrices.items),
        },
    },0, NULL);

    
    bool mouse_locked = false;
    bool wanted_mouse_locked = false;
    bool fullscreen = false;
    bool wanted_fullscreen = false;

    uint32_t imageIndex;
    size_t old_window_width = 0;
    size_t old_window_height = 0;
    Pcs pcs = {
        .view = {
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1,
        },
        .model = {
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1,
        }
    };

    VkSampler depthImagesSampler = {0};
    if (vkCreateSampler(device, &(VkSamplerCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .anisotropyEnable = VK_FALSE,          // depth images do not use anisotropy
        .maxAnisotropy = 1.0f,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
        .unnormalizedCoordinates = VK_FALSE,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .mipLodBias = 0.0f,
        .minLod = 0.0f,
        .maxLod = 0.0f,
    }, NULL, &depthImagesSampler) != VK_SUCCESS) return 1;

    uint64_t lastTime = platform_get_time_milis();

    float time = 0;
    while(platform_still_running()){
        //internals
        uint64_t now = platform_get_time_milis();
        float dt = (now - lastTime) / 1000.0f;  // dt in SECONDS
        lastTime = now;

        if (dt > 0.1f) dt = 0.1f;  // optional dt clamp to avoid big jumps (alt-tab, resize)

        if(wanted_mouse_locked != mouse_locked){
            mouse_locked = wanted_mouse_locked;
            if(mouse_locked) platform_lock_mouse();
            else platform_unlock_mouse();
        }

        if(fullscreen != wanted_fullscreen){
            fullscreen = wanted_fullscreen;
            if(fullscreen) platform_enable_fullscreen();
            else platform_disable_fullscreen();
        }

        platform_window_handle_events();
        if(platform_window_minimized){
            platform_sleep(1);
            continue;
        }

        if(!platform_window_minimized && (old_window_width != swapchainExtent.width || old_window_height != swapchainExtent.height)){
            old_window_width = swapchainExtent.width;
            old_window_height = swapchainExtent.height;

            float fov = 90.0f * PI / 180.0f;
            float zn = 0.001f;
            float zf = 1000.0f;
            float aspect = (float)swapchainExtent.width / (float)swapchainExtent.height;
            float f = 1.0f / tanf(fov * 0.5f);

            pcs.proj[0]  =  f/aspect;
            pcs.proj[1]  = 0;
            pcs.proj[2]  = 0;
            pcs.proj[3]  = 0;

            pcs.proj[4]  = 0;
            pcs.proj[5]  = -f;            // Vulkan: flip Y
            pcs.proj[6]  = 0;
            pcs.proj[7]  = 0;

            pcs.proj[8]  = 0;
            pcs.proj[9]  = 0;
            pcs.proj[10] = zf/(zf-zn);
            pcs.proj[11] = 1;

            pcs.proj[12] = 0;
            pcs.proj[13] = 0;
            pcs.proj[14] = -(zn*zf)/(zf-zn);
            pcs.proj[15] = 0;

            pcs.invProj[0]  = aspect/f;
            pcs.invProj[1]  = 0;
            pcs.invProj[2]  = 0;
            pcs.invProj[3]  = 0;

            pcs.invProj[4]  = 0;
            pcs.invProj[5]  = -1.0f/f;
            pcs.invProj[6]  = 0;
            pcs.invProj[7]  = 0;

            pcs.invProj[8]  = 0;
            pcs.invProj[9]  = 0;
            pcs.invProj[10] = 0;
            pcs.invProj[11] = (zf - zn)/(zn*zf);

            pcs.invProj[12] = 0;
            pcs.invProj[13] = 0;
            pcs.invProj[14] = 1;
            pcs.invProj[15] = -1/f;  // Homogeneous divide
        }

        // logic (update)
        time += dt;

        if(input.keys[KEY_ESCAPE].justPressed) wanted_mouse_locked = !wanted_mouse_locked;
        if(input.keys[KEY_F11].justPressed) wanted_fullscreen = !wanted_fullscreen;

        static float camera_pitch = 0;
        static float camera_yaw = PI;
        static float camera_x = 0;
        static float camera_y = 1;
        static float camera_z = 1.2;
        float mouse_sensitivity = 0.5f; // tweak
        if(mouse_locked){
            camera_yaw   += ((float)input.mouse_x) * mouse_sensitivity * dt;
            camera_pitch += ((float)input.mouse_y) * mouse_sensitivity * dt;
        }
        if(input.keys[KEY_UP].isDown) camera_pitch -= mouse_sensitivity * 5 * dt;
        if(input.keys[KEY_DOWN].isDown) camera_pitch += mouse_sensitivity * 5 * dt;

        if(input.keys[KEY_LEFT].isDown) camera_yaw -= mouse_sensitivity * 5 * dt;
        if(input.keys[KEY_RIGHT].isDown) camera_yaw += mouse_sensitivity * 5 * dt;
        
        if(camera_pitch > PI/2) camera_pitch = PI/2;
        if(camera_pitch < -PI/2) camera_pitch = -PI/2;

        float move_dir_x = 0;
        float move_dir_y = 0;
        float move_dir_z = 0;

        if(input.keys[KEY_SPACE].isDown) move_dir_y += 1;
        if(input.keys[KEY_SHIFT].isDown) move_dir_y -= 1;

        if(input.keys[KEY_W].isDown) {
            move_dir_x += sin(camera_yaw);
            move_dir_z += cos(camera_yaw);
        }

        if(input.keys[KEY_S].isDown) {
            move_dir_x -= sin(camera_yaw);
            move_dir_z -= cos(camera_yaw);
        }

        if(input.keys[KEY_D].isDown) {
            move_dir_x += cos(camera_yaw);
            move_dir_z += -sin(camera_yaw);
        }

        if(input.keys[KEY_A].isDown) {
            move_dir_x -= cos(camera_yaw);
            move_dir_z -= -sin(camera_yaw);
        }

        if(move_dir_x != 0 || move_dir_z != 0){
            float move_dir_mag = sqrtf(move_dir_x * move_dir_x + move_dir_z * move_dir_z);
            move_dir_x /= move_dir_mag;
            move_dir_z /= move_dir_mag;
        }
        
        float move_speed = 2.0f; // units per second
        camera_x += move_dir_x * move_speed * dt;
        camera_y += move_dir_y * move_speed * dt;
        camera_z += move_dir_z * move_speed * dt;

        float translation[] = {
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            -camera_x,-camera_y,-camera_z,1,
        };

        float yaw[] = {
            cos(camera_yaw),0,sin(camera_yaw),0,
            0,1,0,0,
            -sin(camera_yaw),0,cos(camera_yaw),0,
            0,0,0,1,
        };

        float pitch[] = {
            1,0,0,0,
            0,cos(camera_pitch),-sin(camera_pitch),0,
            0,sin(camera_pitch),cos(camera_pitch),0,
            0,0,0,1,
        };

        float view[16];
        mat4_mul(yaw,pitch,view);
        mat4_mul(translation, view,view);
        memcpy(pcs.view, &view, sizeof(view));

        sample_animation(anim, time, 1); // load keyframe
        for(size_t i = 0; i < inverse_bind_matrices.count; i++){
            Mat4 joint_anim_mat4 = {0};
            cgltf_node_transform_world(skin->joints[i], joint_anim_mat4.data);
            mat4_mul(inverse_bind_matrices.items[i].data, joint_anim_mat4.data, combined_joint_matrices.items[i].data);
        }
        //rendering

        vkWaitForFences(device, 1, &renderingFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &renderingFence);
        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, swapchainHasImageSemaphore, NULL, &imageIndex);

        memcpy(jointsBufferMapped, combined_joint_matrices.items, combined_joint_matrices.count*sizeof(*combined_joint_matrices.items));

        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO});
        
        vkCmdTransitionImage(cmd, swapchainImages.items[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        vkCmdBeginRenderingEX(cmd,
            .colorAttachment = swapchainImageViews.items[imageIndex],
            .depthAttachment = swapchainDepthImageViews.items[imageIndex],
            .clearColor = COL_HEX(0xFF181818),
            .renderArea = (
                (VkExtent2D){.width = swapchainExtent.width, .height = swapchainExtent.height}
            )
        );

        vkCmdSetViewport(cmd, 0, 1, &(VkViewport){
            .width = swapchainExtent.width,
            .height = swapchainExtent.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f
        });
            
        vkCmdSetScissor(cmd, 0, 1, &(VkRect2D){
            .extent.width = swapchainExtent.width,
            .extent.height = swapchainExtent.height,
        });

        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &(VkDeviceSize){0});
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pcs), &pcs);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &jointsDescriptorSet, 0, NULL);
        vkCmdDrawIndexed(cmd, indices.count, 1, 0, 0, 0);

        vkCmdEndRendering(cmd);

        vkCmdTransitionImage(cmd, swapchainImages.items[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        vkEndCommandBuffer(cmd);

        vkQueueSubmit(graphicsQueue, 1, &(VkSubmitInfo){
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            
            .waitSemaphoreCount = 1,
            .pWaitDstStageMask = &(VkPipelineStageFlags){VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
            .pWaitSemaphores = &swapchainHasImageSemaphore,

            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &readyToSwapYourChainSemaphore,
        }, renderingFence);

        vkQueuePresentKHR(presentQueue, &(VkPresentInfoKHR){
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &readyToSwapYourChainSemaphore,

            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &imageIndex
        });

        static const double target_frametime = 1.0 / 60.0;   // 60 FPS
        static uint64_t last_frame_end = 0;

        uint64_t frame_end = platform_get_time_milis();
        double frame_dt = (frame_end - now) / 1000.0;        // now = frame start

        double sleep_time = target_frametime - frame_dt;

        if (sleep_time > 0.0)
            platform_sleep((uint32_t)(sleep_time * 1000.0));

        last_frame_end = frame_end;
    }

    return 0;
}